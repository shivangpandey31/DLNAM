# train.py

import math
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PoissonLoss(nn.Module):
    def forward(self, mu, y):
        return torch.mean(mu - y * torch.log(mu + 1e-8))


class Trainer:
    def __init__(self, model_class, **model_kwargs):
        self.device = model_kwargs.get('device', torch.device("cpu"))
        self.num_models = model_kwargs.get('num_models', 3)
        self.exposure_lags = model_kwargs.get('exposure_dims', [30])
        self.output_penalty_lambda = model_kwargs.get('output_penalty', 0)
        self.use_compile = model_kwargs.get('use_compile', False)

        self.ensemble = [
            model_class(**model_kwargs).to(self.device)
            for _ in range(self.num_models)
        ]

        if self.use_compile and hasattr(torch, 'compile'):
            if platform.system() == 'Windows':
                print("  torch.compile() skipped — not supported on Windows.")
            else:
                self.ensemble = [torch.compile(m) for m in self.ensemble]
                print(f"  torch.compile() applied to {self.num_models} models.")

    def train(self, X_exposures, X_c, X_time, Y,
              epochs=1000,
              loss_type='Poisson',
              batch_fraction=None,
              lr_schedule='cosine',
              lr_min=1e-6,
              lr_plateau_factor=0.5,
              lr_plateau_patience=100,
              optim_kwargs=None,
              x_encodings=None,
              processor=None):
        """
        processor : DLNAMDataProcessor (optional)
            When provided, its scaling_types dict is used to automatically
            construct scaling-aware penalty grids.  The grids span the same
            range as the actual training inputs:
              exposure -> processor.scaling_types['exposure']
              conf     -> processor.scaling_types['conf']
              trend    -> 'minmax' (time is always [0,1])
              lag      -> 'minmax' (lag grid is always [0,1])
            If processor is None and output_penalty > 0, the grids default
            to 'minmax' for all inputs.

        x_encodings : list of LongTensors, one per categorical encoding,
                      in the same order as encoding_configs. Or None.
        """
        if optim_kwargs is None:
            optim_kwargs = {'lr': 0.0003, 'weight_decay': 0}

        n_obs = Y.shape[0]

        if batch_fraction is not None:
            if not (0.0 < batch_fraction <= 1.0):
                raise ValueError(
                    f'batch_fraction must be in (0, 1], got {batch_fraction}'
                )
            batch_size    = max(1, round(batch_fraction * n_obs))
            use_minibatch = batch_size < n_obs
            print(
                f'  batch_fraction={batch_fraction:.2f} -> '
                f'batch_size={batch_size} '
                f'({math.ceil(n_obs / batch_size)} steps/epoch)'
            )
        else:
            batch_size    = n_obs
            use_minibatch = False

        if not use_minibatch:
            X_c         = X_c.to(self.device)
            X_time      = X_time.to(self.device)
            Y           = Y.to(self.device)
            X_exposures = [x.to(self.device) for x in X_exposures]
            if x_encodings is not None:
                x_encodings = [e.to(self.device) for e in x_encodings]

        loader     = None
        lag_counts = [x.shape[1] for x in X_exposures]

        if use_minibatch:
            X_exp_concat = torch.cat(X_exposures, dim=1)
            tensors = [X_exp_concat, X_c, X_time, Y]
            if x_encodings is not None:
                for enc in x_encodings:
                    tensors.append(enc.unsqueeze(1).float())
            dataset      = TensorDataset(*tensors)
            safe_workers = 0 if platform.system() == 'Windows' else 4
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=safe_workers,
                pin_memory=(self.device.type == 'cuda'),
            )

        # ------------------------------------------------------------------
        # Bias initialisation
        # ------------------------------------------------------------------
        target_mean  = Y.mean().item()
        initial_bias = math.log(max(target_mean, 1e-8))
        for model in self.ensemble:
            with torch.no_grad():
                model.bias.fill_(initial_bias)

        criterion = PoissonLoss()

        # ------------------------------------------------------------------
        # Precompute penalty grids — automatically scaled from processor
        # ------------------------------------------------------------------
        def _grid_range(scale):
            return (-3.0, 3.0) if scale == 'zscore' else (0.0, 1.0)

        def _make_grid_1d(scale, n=20):
            lo, hi = _grid_range(scale)
            return torch.linspace(lo, hi, n, device=self.device).unsqueeze(1)

        def _make_grid_2d(scale_v, scale_l, n=20):
            lo_v, hi_v = _grid_range(scale_v)
            lo_l, hi_l = _grid_range(scale_l)
            v = torch.linspace(lo_v, hi_v, n, device=self.device)
            l = torch.linspace(lo_l, hi_l, n, device=self.device)
            v_g, l_g = torch.meshgrid(v, l, indexing='ij')
            return torch.stack([v_g.flatten(), l_g.flatten()], dim=1)

        penalty_grids = None
        if self.output_penalty_lambda > 0:
            # Read scaling types from processor if available; default 'minmax'
            if processor is not None and hasattr(processor, 'scaling_types'):
                st = processor.scaling_types
            else:
                st = {}
            sc_exp   = st.get('exposure', 'minmax')
            sc_lag   = st.get('lag',      'minmax')  # always minmax
            sc_conf  = st.get('conf',     'zscore')
            sc_trend = st.get('trend',    'minmax')  # always minmax

            penalty_grids = {
                'surface': _make_grid_2d(sc_exp, sc_lag),
                'conf':    _make_grid_1d(sc_conf),
                'trend':   _make_grid_1d(sc_trend),
            }
            print(
                f"  Penalty grids: exposure={sc_exp}, lag={sc_lag}, "
                f"conf={sc_conf}, trend={sc_trend}"
            )

        # ------------------------------------------------------------------
        # Outer loop
        # ------------------------------------------------------------------
        self.loss_history = []   # reset before new training run

        for i, model in enumerate(self.ensemble):
            label  = f" Training Ensemble {i + 1}/{self.num_models} "
            dashes = 60 - len(label)
            print(f"\n{'-' * (dashes // 2)}{label}{'-' * (dashes - dashes // 2)}")
            optimizer = optim.AdamW(model.parameters(), **optim_kwargs)
            model_loss_history = []   # (epoch, full_dataset_loss) pairs

            if lr_schedule == 'cosine':
                steps_per_epoch = math.ceil(n_obs / batch_size) if use_minibatch else 1
                t_max = max(epochs * steps_per_epoch, 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=lr_min,
                )
            elif lr_schedule == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min',
                    factor=lr_plateau_factor,
                    patience=lr_plateau_patience,
                    min_lr=lr_min,
                )
            else:
                scheduler = None

            for epoch in range(epochs + 1):

                # ----------------------------------------------------------
                # FULL-BATCH
                # ----------------------------------------------------------
                if not use_minibatch:
                    model.train()
                    optimizer.zero_grad()

                    mu           = model(X_exposures, X_c, X_time, x_encodings)
                    poisson_loss = criterion(mu, Y)
                    l2_penalty   = self._penalty(model, penalty_grids)
                    loss         = poisson_loss + self.output_penalty_lambda * l2_penalty

                    if torch.isnan(loss):
                        print(f"  NaN loss at epoch {epoch} – stopping early.")
                        break

                    loss.backward()
                    optimizer.step()
                    if scheduler is not None and lr_schedule == 'cosine':
                        scheduler.step()

                    loss_for_print = loss.item()

                # ----------------------------------------------------------
                # MINIBATCH
                # ----------------------------------------------------------
                else:
                    model.train()
                    epoch_loss = 0.0
                    n_batches  = 0
                    nan_hit    = False

                    for batch in loader:
                        n_enc = len(x_encodings) if x_encodings is not None else 0
                        X_exp_b, X_c_b, X_time_b, Y_b = batch[:4]
                        enc_bs = [
                            batch[4 + j].squeeze(1).long().to(
                                self.device, non_blocking=True
                            )
                            for j in range(n_enc)
                        ] if n_enc > 0 else None

                        X_exp_b  = X_exp_b.to(self.device, non_blocking=True)
                        X_c_b    = X_c_b.to(self.device, non_blocking=True)
                        X_time_b = X_time_b.to(self.device, non_blocking=True)
                        Y_b      = Y_b.to(self.device, non_blocking=True)
                        X_exp_split = list(torch.split(X_exp_b, lag_counts, dim=1))

                        optimizer.zero_grad()

                        mu_b         = model(X_exp_split, X_c_b, X_time_b, enc_bs)
                        poisson_loss = criterion(mu_b, Y_b)
                        l2_penalty   = self._penalty(model, penalty_grids)
                        loss         = poisson_loss + self.output_penalty_lambda * l2_penalty

                        if torch.isnan(loss):
                            print(f"  NaN loss at epoch {epoch} – stopping early.")
                            nan_hit = True
                            break

                        loss.backward()
                        optimizer.step()
                        if scheduler is not None and lr_schedule == 'cosine':
                            scheduler.step()

                        epoch_loss += loss.item()
                        n_batches  += 1

                    if nan_hit:
                        break

                    loss_for_print = epoch_loss / max(n_batches, 1)

                # Record every epoch for training curve plot
                model_loss_history.append((epoch, loss_for_print))

                # ----------------------------------------------------------
                # Plateau — step every epoch on full-dataset loss
                # ----------------------------------------------------------
                if lr_schedule == 'plateau' and scheduler is not None:
                    model.eval()
                    if use_minibatch:
                        X_exp_d = [x.to(self.device) for x in X_exposures]
                        X_c_d, X_t_d = X_c.to(self.device), X_time.to(self.device)
                        Y_d   = Y.to(self.device)
                        enc_d = [e.to(self.device) for e in x_encodings] if x_encodings is not None else None
                    else:
                        X_exp_d, X_c_d, X_t_d, Y_d = (
                            X_exposures, X_c, X_time, Y
                        )
                        enc_d = x_encodings
                    with torch.no_grad():
                        mu_p = model(X_exp_d, X_c_d, X_t_d, enc_d)
                        fl_p = criterion(mu_p, Y_d).item()
                    scheduler.step(fl_p)

                # ----------------------------------------------------------
                # Diagnostics
                # ----------------------------------------------------------
                if epoch % max(epochs // 10, 1) == 0:
                    model.eval()
                    if use_minibatch:
                        X_exp_d = [x.to(self.device) for x in X_exposures]
                        X_c_d, X_t_d = X_c.to(self.device), X_time.to(self.device)
                        Y_d   = Y.to(self.device)
                        enc_d = [e.to(self.device) for e in x_encodings] if x_encodings is not None else None
                    else:
                        X_exp_d, X_c_d, X_t_d, Y_d = (
                            X_exposures, X_c, X_time, Y
                        )
                        enc_d = x_encodings
                    with torch.no_grad():
                        mu_e      = model(X_exp_d, X_c_d, X_t_d, enc_d)
                        phi       = torch.mean(
                            ((Y_d - mu_e) ** 2) / (mu_e + 1e-8)
                        ).item()
                        full_loss = criterion(mu_e, Y_d).item()

                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"  Epoch {epoch:4d} | Loss: {full_loss:9.4f}"
                        f" | Phi: {phi:7.3f}"
                        f" | LR: {current_lr:.2e}"
                    )

            self.loss_history.append(model_loss_history)

    # ------------------------------------------------------------------
    # Output penalty — all network types with scaling-aware grids
    # ------------------------------------------------------------------
    def _penalty(self, model, penalty_grids):
        """
        Evaluates L2 output penalty for every network component:
          surface subnets    — mean squared output over 2D grid (exposure × lag)
          confounder subnets — mean squared output over 1D grid (confounder value)
          trend subnets      — mean squared output over 1D grid (normalised time)
          encodings          — mean squared embedding weights directly
                               (no grid needed: weights are the outputs)
        """
        if self.output_penalty_lambda == 0 or penalty_grids is None:
            return 0.0

        l2 = 0.0
        for subnets in model.surface_subnets:
            for s_net in subnets:
                l2 = l2 + torch.mean(s_net(penalty_grids['surface']) ** 2)
        for subnets in model.conf_subnets:
            for s_net in subnets:
                l2 = l2 + torch.mean(s_net(penalty_grids['conf']) ** 2)
        for s_net in model.trend_subnets:
            l2 = l2 + torch.mean(s_net(penalty_grids['trend']) ** 2)
        # Encoding: penalise all parameters in the encoding network
        for enc in model.encodings:
            for p in enc.net.parameters():
                l2 = l2 + torch.mean(p ** 2)
        return l2
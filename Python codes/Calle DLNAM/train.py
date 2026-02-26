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
        self.output_penalty_lambda = model_kwargs.get('output_penalty', 1e-4)

        self.use_compile = model_kwargs.get('use_compile', False)

        self.ensemble = [
            model_class(**model_kwargs).to(self.device)
            for _ in range(self.num_models)
        ]

        # torch.compile() traces the model's computation graph and emits
        # optimised machine code for the target hardware.  Operations such
        # as linear + activation are fused into single GPU kernels, reducing
        # memory bandwidth pressure.  The first forward pass is slow (compile
        # happens then); all subsequent passes use the compiled version.
        #
        # Only applied when use_compile=True AND PyTorch >= 2.0 is available.
        # Disabled automatically on Windows due to known compiler limitations.
        if self.use_compile and hasattr(torch, 'compile'):
            if platform.system() == 'Windows':
                print("  torch.compile() skipped — not supported on Windows.")
            else:
                self.ensemble = [
                    torch.compile(m) for m in self.ensemble
                ]
                print(f"  torch.compile() applied to {self.num_models} models.")

    def train(self, X_exposures, X_c, X_time, Y,
              epochs=1000,
              loss_type='Poisson',
              batch_fraction=None,
              lr_schedule=True,
              optim_kwargs=None):
        """
        batch_fraction : float in (0, 1] or None
            None  -> full-batch gradient descent (default).
                     The entire dataset is used per gradient step.

            float -> minibatch training. Batch size is computed as
                     round(batch_fraction * n_obs), scaling automatically
                     when you move to larger datasets.

                     Recommended values on Chicago (~4600 obs):
                       0.20 -> ~920 obs  (~5  steps/epoch)
                       0.10 -> ~460 obs  (~10 steps/epoch)
                       0.05 -> ~230 obs  (~20 steps/epoch)

                     Portable across dataset sizes: 0.10 always means
                     10 percent of whatever dataset you pass in.
        """
        if optim_kwargs is None:
            optim_kwargs = {'lr': 0.0003, 'weight_decay': 1e-3}

        # ------------------------------------------------------------------
        # Data placement:
        #   Full-batch  -> move everything to device once, reuse each epoch.
        #   Minibatch   -> keep on CPU so DataLoader can pin memory.
        #                  Moving to GPU before pinning causes:
        #                  'cannot pin torch.cuda.FloatTensor'
        #                  Each batch is moved to device inside the loop.
        # ------------------------------------------------------------------
        n_obs = Y.shape[0]

        # Resolve batch_fraction to a concrete batch size.
        # batch_fraction=None  -> full-batch
        # batch_fraction=0.10  -> 10 percent of dataset per batch
        # Clamped to [1, n_obs] to guard against extreme values.
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

        # ------------------------------------------------------------------
        # Minibatch setup — only runs when use_minibatch is True.
        # Completely skipped in full-batch mode (zero overhead).
        #
        # X_exposures is concatenated along dim=1 for DataLoader and
        # re-split by original lag counts inside each batch iteration.
        # ------------------------------------------------------------------
        loader     = None
        lag_counts = [x.shape[1] for x in X_exposures]

        if use_minibatch:
            X_exp_concat = torch.cat(X_exposures, dim=1)  # (N, total_lag_cols)
            dataset = TensorDataset(X_exp_concat, X_c, X_time, Y)
            # num_workers > 0 spawns background processes that prepare the
            # next batch while the GPU processes the current one, eliminating
            # the data-loading bottleneck on large datasets.
            # pin_memory=True stores batches in page-locked memory for faster
            # CPU -> GPU transfers.
            # On Windows multiprocessing requires a __main__ guard so we
            # fall back to 0 workers automatically.
            safe_workers = 0 if platform.system() == 'Windows' else 4
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=safe_workers,
                pin_memory=(self.device.type == 'cuda'),
            )

        # ------------------------------------------------------------------
        # Bias initialisation: log(target_mean)
        # ------------------------------------------------------------------
        target_mean  = Y.mean().item()
        initial_bias = math.log(max(target_mean, 1e-8))
        for model in self.ensemble:
            with torch.no_grad():
                model.bias.fill_(initial_bias)

        criterion = PoissonLoss()

        # ------------------------------------------------------------------
        # Precompute output-penalty grid once — does not depend on weights
        # ------------------------------------------------------------------
        penalty_grid = None
        if self.output_penalty_lambda > 0:
            grid_size    = 20
            v            = torch.linspace(0.0, 1.0, grid_size, device=self.device)
            l            = torch.linspace(0.0, 1.0, grid_size, device=self.device)
            v_g, l_g     = torch.meshgrid(v, l, indexing='ij')
            penalty_grid = torch.stack(
                [v_g.flatten(), l_g.flatten()], dim=1
            )  # (grid_size^2, 2)

        # ------------------------------------------------------------------
        # Outer loop: one independent training run per ensemble member
        # ------------------------------------------------------------------
        for i, model in enumerate(self.ensemble):
            print(f"\n--- Training Ensemble {i + 1}/{self.num_models} ---")
            optimizer = optim.AdamW(model.parameters(), **optim_kwargs)

            # Cosine annealing scheduler — decays lr from initial value to
            # lr_min over the full training run.  T_max is set to the total
            # number of gradient steps (not epochs) so the decay is smooth
            # regardless of whether full-batch or minibatch is used.
            #
            # With full-batch:   1 step per epoch  -> T_max = epochs
            # With minibatch:    k steps per epoch -> T_max = epochs * k
            #   where k = ceil(n_obs / batch_size)
            #
            # This ensures the lr reaches its minimum at the final update
            # in both cases, giving a consistent decay shape.
            if lr_schedule:
                if use_minibatch:
                    steps_per_epoch = math.ceil(n_obs / batch_size)
                else:
                    steps_per_epoch = 1
                t_max = max(epochs * steps_per_epoch, 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=t_max,
                    eta_min=1e-6,
                )
            else:
                scheduler = None

            for epoch in range(epochs + 1):

                # ----------------------------------------------------------
                # FULL-BATCH path
                # ----------------------------------------------------------
                if not use_minibatch:
                    model.train()
                    optimizer.zero_grad()

                    mu           = model(X_exposures, X_c, X_time)
                    poisson_loss = criterion(mu, Y)
                    l2_penalty   = self._penalty(model, penalty_grid)
                    loss         = (poisson_loss
                                    + self.output_penalty_lambda * l2_penalty)

                    if torch.isnan(loss):
                        print(f"  NaN loss at epoch {epoch} – stopping early.")
                        break

                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    loss_for_print = loss.item()

                # ----------------------------------------------------------
                # MINIBATCH path
                # ----------------------------------------------------------
                else:
                    model.train()
                    epoch_loss = 0.0
                    n_batches  = 0
                    nan_hit    = False

                    for X_exp_b, X_c_b, X_time_b, Y_b in loader:
                        # Transfer batch from CPU to GPU.
                        # non_blocking=True overlaps transfer with GPU compute.
                        X_exp_b  = X_exp_b.to(self.device, non_blocking=True)
                        X_c_b    = X_c_b.to(self.device, non_blocking=True)
                        X_time_b = X_time_b.to(self.device, non_blocking=True)
                        Y_b      = Y_b.to(self.device, non_blocking=True)
                        X_exp_split = list(
                            torch.split(X_exp_b, lag_counts, dim=1)
                        )

                        optimizer.zero_grad()

                        mu_b         = model(X_exp_split, X_c_b, X_time_b)
                        poisson_loss = criterion(mu_b, Y_b)
                        l2_penalty   = self._penalty(model, penalty_grid)
                        loss         = (poisson_loss
                                        + self.output_penalty_lambda * l2_penalty)

                        if torch.isnan(loss):
                            print(
                                f"  NaN loss at epoch {epoch} – stopping early."
                            )
                            nan_hit = True
                            break

                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                        epoch_loss += loss.item()
                        n_batches  += 1

                    if nan_hit:
                        break

                    loss_for_print = epoch_loss / max(n_batches, 1)

                # ----------------------------------------------------------
                # Periodic diagnostics — always on the full dataset
                # ----------------------------------------------------------
                if epoch % max(epochs // 10, 1) == 0:
                    model.eval()
                    # For diagnostics in minibatch mode, move full data to
                    # device temporarily — it lives on CPU otherwise.
                    if use_minibatch:
                        X_exp_diag = [x.to(self.device) for x in X_exposures]
                        X_c_diag   = X_c.to(self.device)
                        X_time_diag = X_time.to(self.device)
                        Y_diag     = Y.to(self.device)
                    else:
                        X_exp_diag  = X_exposures
                        X_c_diag    = X_c
                        X_time_diag = X_time
                        Y_diag      = Y
                    with torch.no_grad():
                        mu_e = model(X_exp_diag, X_c_diag, X_time_diag)
                        phi  = torch.mean(
                            ((Y_diag - mu_e) ** 2) / (mu_e + 1e-8)
                        ).item()
                    # Also report current lr so decay is visible in output
                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"  Epoch {epoch:4d} | Loss: {loss_for_print:.4f}"
                        f" | Phi: {phi:.3f}"
                        f" | LR: {current_lr:.2e}"
                    )

    # ------------------------------------------------------------------
    # Output penalty helper — shared by both training paths
    # ------------------------------------------------------------------
    def _penalty(self, model, penalty_grid):
        if self.output_penalty_lambda == 0 or penalty_grid is None:
            return 0.0
        l2 = 0.0
        for subnets in model.surface_subnets:
            for s_net in subnets:
                l2 = l2 + torch.mean(s_net(penalty_grid) ** 2)
        return l2
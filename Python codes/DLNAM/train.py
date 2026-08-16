"""
train.py - vectorised training for DLNAM ensembles.

All ensemble members train in one pass via torch.func
(stack_module_state + functional_call + vmap). The trainer supports full-batch
or minibatch optimisation, link-matched loss, cosine scheduling, gradient
clipping, finite-loss guards, and optional early stopping with best-weight
restoration.

Early stopping monitors the mean objective over the complete training epoch by
default. If ``validation_inputs`` and ``validation_y`` are supplied to ``fit``,
it instead monitors the validation objective. ``patience`` therefore counts
EPOCHS, not occasional diagnostic-print intervals.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Optional

import torch
from torch.func import stack_module_state, functional_call, vmap

from .config import ModelConfig, TrainConfig, CategoricalTermSpec
from .model import DLNAM


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _progress_bar(progress: float, width: int = 20) -> str:
    """Return a compact fixed-width Unicode progress bar."""
    progress = min(max(float(progress), 0.0), 1.0)
    filled = int(round(progress * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"



def _per_member_nll(preds: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    """preds: (M, B, 1), y: (B, 1) -> (M,) per-member mean NLL."""
    yb = y.unsqueeze(0)
    if loss == "bernoulli":
        mu = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        elem = -yb * torch.log(mu) - (1.0 - yb) * torch.log(1.0 - mu)
    elif loss == "poisson":
        elem = preds - yb * torch.log(preds + 1e-8)
    elif loss == "gaussian":
        elem = (preds - yb) ** 2
    else:
        raise ValueError(f"Unknown loss: {loss}")
    return elem.mean(dim=(1, 2))


def _per_member_profiled_poisson_nll(
    eta: torch.Tensor,
    y: torch.Tensor,
    group_index: torch.Tensor,
) -> torch.Tensor:
    """Profile Poisson nuisance intercepts out of complete strata.

    Parameters
    ----------
    eta
        Linear predictors with shape ``(M, B, 1)``.
    y
        Non-negative outcomes with shape ``(B, 1)``.
    group_index
        Local contiguous group IDs ``0..G-1`` for the B rows. Every stratum in
        this mini-batch must be complete.

    Returns
    -------
    torch.Tensor, shape (M,)
        Mean profiled negative log-likelihood per stratum, with terms constant
        in the neural parameters omitted.

    Notes
    -----
    For stratum g with Y_g=sum_j y_gj, profiling the nuisance intercept gives

        -sum_j y_gj * eta_gj + Y_g * logsumexp_j(eta_gj)

    up to a parameter-independent constant. With one event per matched set this
    is exactly the within-stratum softmax/cross-entropy objective.
    """
    if eta.ndim != 3 or eta.shape[-1] != 1:
        raise ValueError("eta must have shape (ensemble, rows, 1)")
    if y.ndim != 2 or y.shape[-1] != 1:
        raise ValueError("y must have shape (rows, 1)")
    if group_index.ndim != 1 or group_index.shape[0] != y.shape[0]:
        raise ValueError("group_index must be a 1-D tensor aligned with y")
    if torch.any(y < 0):
        raise ValueError("profiled Poisson requires non-negative outcomes")

    e = eta.squeeze(-1)                     # (M, B)
    yy = y.squeeze(-1)                      # (B,)
    g = group_index.to(dtype=torch.long)
    G = int(g.max().item()) + 1 if g.numel() else 0
    if G == 0:
        raise ValueError("profiled Poisson batch contains no strata")

    # Outcome totals are common across ensemble members.
    y_tot = torch.zeros(G, device=y.device, dtype=y.dtype)
    y_tot.scatter_add_(0, g, yy)
    if torch.any(y_tot <= 0):
        raise ValueError(
            "every elimination stratum must contain positive outcome mass; "
            "for a matched case-crossover design this normally means one event "
            "in each stratum"
        )

    gi = g.unsqueeze(0).expand(e.shape[0], -1)
    neg_inf = torch.finfo(e.dtype).min
    group_max = torch.full(
        (e.shape[0], G), neg_inf, device=e.device, dtype=e.dtype
    )
    group_max.scatter_reduce_(1, gi, e, reduce="amax", include_self=True)

    centered = e - group_max.gather(1, gi)
    exp_sum = torch.zeros_like(group_max)
    exp_sum.scatter_add_(1, gi, torch.exp(centered))
    log_denom = group_max + torch.log(exp_sum.clamp_min(1e-30))

    observed_score = (e * yy.unsqueeze(0)).sum(dim=1)
    normalizer = (log_denom * y_tot.unsqueeze(0)).sum(dim=1)
    return (normalizer - observed_score) / float(G)


class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig,
                 device: Optional[torch.device] = None):
        self.cfg = model_config
        self.tcfg = train_config
        self._validate_link_loss(model_config, train_config)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(train_config.seed)
        self.ensemble = [
            DLNAM.from_config(model_config).to(self.device)
            for _ in range(train_config.n_ensemble)
        ]
        self.loss_history: list[list[tuple]] = []
        self.monitor_history: list[tuple[int, float]] = []
        self.fit_summary: dict = {}

    @staticmethod
    def _validate_link_loss(model_config: ModelConfig, train_config: TrainConfig) -> None:
        valid = {
            ("log", "poisson"),
            ("log", "profiled_poisson"),
            ("logit", "bernoulli"),
            ("identity", "gaussian"),
        }
        pair = (model_config.link, train_config.loss)
        if pair not in valid:
            allowed = ", ".join(f"{link}/{loss}" for link, loss in sorted(valid))
            raise ValueError(
                f"unsupported link/loss pair {pair[0]}/{pair[1]}; "
                f"supported pairs are {allowed}"
            )

    def fit(self, inputs: dict, y: torch.Tensor, *,
            eliminate_index: Optional[torch.Tensor] = None,
            validation_inputs: Optional[dict] = None,
            validation_y: Optional[torch.Tensor] = None,
            validation_eliminate_index: Optional[torch.Tensor] = None):
        """Fit the vectorised ensemble.

        Supplying validation data changes only the early-stopping monitor; model
        updates still use ``inputs``/``y``. Validation inputs must already be on
        model scale (normally produced by the same fitted DataProcessor/scalers).
        """
        if (validation_inputs is None) != (validation_y is None):
            raise ValueError("validation_inputs and validation_y must be supplied together")

        profiled = self.tcfg.loss == "profiled_poisson"
        if profiled and eliminate_index is None:
            raise ValueError(
                "loss='profiled_poisson' requires eliminate_index. Prepare data "
                "with DataProcessor.prepare(..., eliminate_col='your_stratum_col') "
                "and pass prepared.eliminate_index to Trainer.fit()."
            )
        if not profiled and eliminate_index is not None:
            raise ValueError(
                "eliminate_index is only used with loss='profiled_poisson'"
            )
        if validation_inputs is not None:
            if profiled and validation_eliminate_index is None:
                raise ValueError(
                    "profiled validation data require validation_eliminate_index"
                )
            if not profiled and validation_eliminate_index is not None:
                raise ValueError(
                    "validation_eliminate_index is only used with profiled_poisson"
                )

        if profiled:
            learned_strata = [
                name for name, spec in self.cfg.terms.items()
                if isinstance(spec, CategoricalTermSpec) and spec.role == "strata"
            ]
            if learned_strata:
                raise ValueError(
                    "profiled_poisson eliminates nuisance strata analytically and "
                    "must not be combined with learned strata terms. Remove/disable "
                    f"strata_config (found: {learned_strata})."
                )

        t = self.tcfg
        start = time.perf_counter()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        y = y.to(self.device)
        if eliminate_index is not None:
            eliminate_index = eliminate_index.to(self.device, dtype=torch.long)
            eliminate_index = self._canonical_group_index(eliminate_index)
            self._validate_profiled_groups(y, eliminate_index)
        if validation_inputs is not None:
            validation_inputs = {k: v.to(self.device) for k, v in validation_inputs.items()}
            validation_y = validation_y.to(self.device)
            if validation_eliminate_index is not None:
                validation_eliminate_index = validation_eliminate_index.to(
                    self.device, dtype=torch.long
                )
                validation_eliminate_index = self._canonical_group_index(
                    validation_eliminate_index
                )
                self._validate_profiled_groups(
                    validation_y, validation_eliminate_index
                )

        if t.show_progress:
            if profiled:
                n_groups = int(eliminate_index.max().item()) + 1
                bs, n_steps = self._batch_plan(n_groups)
                batch_label = "strata/batch"
            else:
                bs, n_steps = self._batch_plan(int(y.shape[0]))
                batch_label = "batch size"
            print("\nDLNAM training")
            print(f"  device:       {self.device}")
            if self.device.type == "cuda":
                print(f"  GPU:          {torch.cuda.get_device_name(self.device)}")
            print(f"  samples:      {int(y.shape[0]):,}")
            print(f"  ensemble:     {len(self.ensemble)}")
            if profiled:
                print(f"  likelihood:   profiled Poisson (stratum intercepts eliminated)")
                print(f"  strata:       {n_groups:,}")
            print(f"  {batch_label+':':14s}{bs:,} ({n_steps} steps/epoch)")
            print(f"  epochs:       {t.epochs}")

        y_mean = float(y.mean())
        for m in self.ensemble:
            if profiled:
                # A global intercept is non-identifiable after profiling the
                # stratum intercepts; set it to zero and leave it unoptimised.
                with torch.no_grad():
                    m.intercept.zero_()
            else:
                m.init_intercept(y_mean)
            m.train()

        run = self._fit_vectorised(
            inputs, y,
            eliminate_index=eliminate_index,
            validation_inputs=validation_inputs,
            validation_y=validation_y,
            validation_eliminate_index=validation_eliminate_index,
        )
        for m in self.ensemble:
            m.eval()

        elapsed = time.perf_counter() - start
        self.fit_summary = {
            "fit_seconds": float(elapsed),
            "epochs_requested": int(t.epochs),
            "epochs_completed": int(run["epochs_completed"]),
            "n_samples": int(y.shape[0]),
            "n_ensemble": int(len(self.ensemble)),
            "device": str(self.device),
            "early_stopping": bool(t.early_stopping),
            "stopped_early": bool(run["stopped_early"]),
            "best_epoch": run["best_epoch"],
            "best_objective": run["best_objective"],
            "monitor": "validation" if validation_inputs is not None else "training_epoch",
            "loss": t.loss,
            "eliminated_strata": (
                None if eliminate_index is None
                else int(eliminate_index.max().item()) + 1
            ),
        }
        if t.show_progress:
            print(f"  elapsed:      {_format_seconds(elapsed)}")
        return self

    # ------------------------------------------------------------------
    # Vectorised ensemble: one vmap pass trains every member
    # ------------------------------------------------------------------
    def _fit_vectorised(self, inputs, y, *, eliminate_index=None,
                        validation_inputs=None, validation_y=None,
                        validation_eliminate_index=None):
        t = self.tcfg
        n = y.shape[0]
        profiled = t.loss == "profiled_poisson"
        base = copy.deepcopy(self.ensemble[0]).to("meta")
        params, buffers = stack_module_state(self.ensemble)
        params = {
            k: v.detach().clone().requires_grad_(not (profiled and k == "intercept"))
            for k, v in params.items()
        }

        def fmodel(p, b, inp):
            return functional_call(
                base,
                (p, b),
                (inp,),
                {"return_penalty": True, "return_eta": profiled},
            )
        batched = vmap(fmodel, in_dims=(0, 0, None), randomness="different")

        strata_names = {
            name for name, spec in self.cfg.terms.items()
            if isinstance(spec, CategoricalTermSpec) and spec.role == "strata"
        }
        strata_prefixes = tuple(f"terms.{name}." for name in strata_names)
        regular_params, strata_params = [], []
        for key, value in params.items():
            if profiled and key == "intercept":
                continue
            if strata_prefixes and key.startswith(strata_prefixes):
                strata_params.append(value)
            else:
                regular_params.append(value)

        param_groups = []
        if regular_params:
            param_groups.append({"params": regular_params, "weight_decay": t.weight_decay})
        if strata_params:
            param_groups.append({"params": strata_params, "weight_decay": t.strata_weight_decay})

        opt = torch.optim.AdamW(param_groups, lr=t.lr)
        if profiled:
            n_groups = int(eliminate_index.max().item()) + 1
            bs, n_steps = self._batch_plan(n_groups)
        else:
            bs, n_steps = self._batch_plan(n)
        # Keep the existing v2 epoch convention (0..epochs inclusive) for
        # backwards compatibility with current experiments.
        sched = self._make_sched(opt, (t.epochs + 1) * n_steps)
        hist = [[] for _ in self.ensemble]
        diagnostics_every = self._diagnostics_interval(t)

        best_monitor = float("inf")
        best_epoch = None
        best_params = None
        best_buffers = None
        epochs_without_improvement = 0
        stopped_early = False
        epochs_completed = 0
        self.monitor_history = []

        abort_training = False
        progress_drawn = False
        if self.device.type == "cuda" and t.gpu_diagnostics:
            torch.cuda.empty_cache()

        for epoch in range(t.epochs + 1):
            epoch_start = time.perf_counter()

            # Reset CUDA peak counters at the start of each epoch so the
            # reported peak is the peak for THIS epoch, not the whole run.
            if self.device.type == "cuda" and t.gpu_diagnostics:
                torch.cuda.reset_peak_memory_stats(self.device)
            if profiled:
                profile_batches = self._profiled_epoch_batches(
                    eliminate_index, groups_per_batch=bs
                )
                perm = None
            else:
                perm = torch.randperm(n, device=self.device) if bs < n else None
                profile_batches = None
            ep_sum = torch.zeros(len(self.ensemble), device=self.device)
            seen = 0

            for s in range(n_steps):
                if profiled:
                    idx, local_groups, groups_in_batch = profile_batches[s]
                    bi = {k: v[idx] for k, v in inputs.items()}
                    by = y[idx]
                else:
                    bi, by = self._batch(inputs, y, perm, s, bs, n)
                opt.zero_grad(set_to_none=True)
                output, penalties = batched(params, buffers, bi)
                if profiled:
                    per_m = _per_member_profiled_poisson_nll(
                        output, by, local_groups
                    ) + penalties
                else:
                    per_m = _per_member_nll(output, by, t.loss) + penalties
                loss = per_m.mean()

                if not torch.isfinite(loss):
                    print(f"  warning: non-finite objective at epoch {epoch}; stopping")
                    stopped_early = True
                    abort_training = True
                    break

                loss.backward()
                if t.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params.values(), t.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()

                if profiled:
                    ep_sum += per_m.detach() * groups_in_batch
                    seen += groups_in_batch
                else:
                    b = int(by.shape[0])
                    ep_sum += per_m.detach() * b
                    seen += b

            if seen == 0:
                break

            ep_loss = ep_sum / seen
            epochs_completed = epoch
            for i in range(len(self.ensemble)):
                hist[i].append((epoch, float(ep_loss[i])))

            if validation_inputs is not None:
                if profiled:
                    monitor_per_member = self._profiled_dataset_objective(
                        batched, params, buffers,
                        validation_inputs, validation_y,
                        validation_eliminate_index, bs,
                    )
                else:
                    monitor_per_member = self._dataset_objective(
                        batched, params, buffers, validation_inputs, validation_y, bs
                    )
            else:
                monitor_per_member = ep_loss
            monitor = float(monitor_per_member.mean())
            self.monitor_history.append((epoch, monitor))

            if t.early_stopping:
                improvement = best_monitor - monitor
                if improvement > t.early_stopping_min_delta:
                    best_monitor = monitor
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_params = {k: v.detach().clone() for k, v in params.items()}
                    best_buffers = {k: v.detach().clone() for k, v in buffers.items()}
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= t.early_stopping_patience:
                    stopped_early = True
                    # The live display leaves the cursor at the end of the
                    # second status line. Finish that line before printing the
                    # stopping message.
                    if t.show_progress and progress_drawn:
                        print()
                    print(
                        f"  early stopping at epoch {epoch}; best epoch {best_epoch}, "
                        f"best objective {best_monitor:.6f}"
                    )
                    break
                
            if (t.show_progress and diagnostics_every and
                    (epoch == 0 or epoch % diagnostics_every == 0 or epoch == t.epochs)):

                lr = opt.param_groups[0]["lr"]
                epoch_time = time.perf_counter() - epoch_start
                progress = min(epoch / max(t.epochs, 1), 1.0)
                bar = _progress_bar(progress, width=20)

                # Optional validation loss
                monitor_text = ""
                if validation_inputs is not None:
                    monitor_text = f" | Val {monitor:.4f}"

                # First live line: training progress. Keep this independent of
                # GPU details so both physical terminal rows remain short.
                epoch_msg = (
                    f"Epoch {epoch:4d}/{t.epochs} "
                    f"{bar} {progress * 100:5.1f}%"
                    f" | Loss {float(ep_loss.mean()):.5f}"
                    f"{monitor_text}"
                    f" | LR {lr:.1e}"
                    f" | {epoch_time:.1f}s"
                )

                # Second live line: GPU diagnostics. ``device_used`` reflects
                # memory occupied on the whole visible CUDA device, while
                # allocated/reserved are this PyTorch process's allocator
                # statistics. Peak values are reset at the start of each epoch.
                if self.device.type == "cuda" and t.gpu_diagnostics:
                    torch.cuda.synchronize(self.device)

                    gib = 1024**3
                    free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                    device_used = (total_bytes - free_bytes) / gib
                    device_total = total_bytes / gib
                    allocated = torch.cuda.memory_allocated(self.device) / gib
                    reserved = torch.cuda.memory_reserved(self.device) / gib
                    peak_allocated = torch.cuda.max_memory_allocated(self.device) / gib
                    peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib

                    gpu_msg = (
                        f"GPU {device_used:.2f}/{device_total:.2f}G"
                        f" | torch {allocated:.2f}G"
                        f" | reserved {reserved:.2f}G"
                        f" | peak {peak_allocated:.2f}/{peak_reserved:.2f}G"
                    )
                else:
                    gpu_msg = "GPU diagnostics unavailable"

                # Draw exactly two live lines. The cursor is deliberately left
                # at the end of line 2. On the next update, move up one row,
                # clear/rewrite line 1, then clear/rewrite line 2.
                if progress_drawn:
                    print("\033[1A\r\033[2K", end="")

                print(epoch_msg)
                print("\r\033[2K" + gpu_msg, end="", flush=True)
                progress_drawn = True

            if abort_training:
                break

        if t.show_progress and progress_drawn:
            print()
        
        use_best = (
            t.early_stopping and t.restore_best_weights and best_params is not None
        )
        if use_best:
            self._writeback(best_params, best_buffers)
        else:
            self._writeback(params, buffers)

        self.loss_history = hist
        return {
            "epochs_completed": epochs_completed,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_objective": None if best_epoch is None else float(best_monitor),
        }

    def _dataset_objective(self, batched, params, buffers, inputs, y, chunk_size):
        """Per-member objective over a dataset without one giant forward pass."""
        n = int(y.shape[0])
        chunk_size = max(1, min(int(chunk_size), n))
        total = torch.zeros(len(self.ensemble), device=self.device)
        seen = 0
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                bi = {k: v[start:stop] for k, v in inputs.items()}
                by = y[start:stop]
                preds, penalties = batched(params, buffers, bi)
                per_m = _per_member_nll(preds, by, self.tcfg.loss) + penalties
                b = stop - start
                total += per_m * b
                seen += b
        return total / max(seen, 1)

    def _profiled_dataset_objective(
        self, batched, params, buffers, inputs, y, eliminate_index,
        groups_per_batch,
    ):
        """Frozen profiled objective, chunked only at complete strata."""
        total = torch.zeros(len(self.ensemble), device=self.device)
        seen_groups = 0
        batches = self._profiled_epoch_batches(
            eliminate_index, groups_per_batch=groups_per_batch
        )
        with torch.no_grad():
            for idx, local_groups, groups_in_batch in batches:
                bi = {k: v[idx] for k, v in inputs.items()}
                by = y[idx]
                eta, penalties = batched(params, buffers, bi)
                per_m = _per_member_profiled_poisson_nll(
                    eta, by, local_groups
                ) + penalties
                total += per_m * groups_in_batch
                seen_groups += groups_in_batch
        return total / max(seen_groups, 1)

    def _writeback(self, params, buffers):
        for i, m in enumerate(self.ensemble):
            sd = {k: v[i].detach() for k, v in params.items()}
            sd.update({k: v[i].detach() for k, v in buffers.items()})
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "vmap writeback key mismatch: trained weights not applied. "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}."
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _diagnostics_interval(t: TrainConfig) -> int:
        if t.diagnostics_every is None:
            return max(1, t.epochs // 10) if t.epochs else 0
        return int(t.diagnostics_every)

    @staticmethod
    def _canonical_group_index(group_index: torch.Tensor) -> torch.Tensor:
        """Map arbitrary integer group labels to contiguous 0..G-1 codes."""
        if group_index.ndim != 1:
            raise ValueError("eliminate_index must be a 1-D tensor")
        if group_index.numel() == 0:
            raise ValueError("eliminate_index is empty")
        _, inverse = torch.unique(group_index, sorted=True, return_inverse=True)
        return inverse.to(dtype=torch.long)

    @staticmethod
    def _validate_profiled_groups(y: torch.Tensor, group_index: torch.Tensor) -> None:
        if group_index.shape[0] != y.shape[0]:
            raise ValueError(
                "eliminate_index must have exactly one group ID per outcome row"
            )
        if torch.any(~torch.isfinite(y)):
            raise ValueError("profiled Poisson outcomes must be finite")
        if torch.any(y < 0):
            raise ValueError("profiled Poisson outcomes must be non-negative")
        G = int(group_index.max().item()) + 1
        totals = torch.zeros(G, device=y.device, dtype=y.dtype)
        totals.scatter_add_(0, group_index, y.reshape(-1))
        bad = torch.nonzero(totals <= 0, as_tuple=False).reshape(-1)
        if bad.numel():
            raise ValueError(
                f"profiled Poisson found {bad.numel():,} strata with no events. "
                "Each matched stratum must contain positive outcome mass; in your "
                "case-crossover data this normally means one event per stratum."
            )

    def _profiled_epoch_batches(self, group_index, *, groups_per_batch):
        """Create mini-batches of COMPLETE elimination strata.

        ``batch_fraction`` therefore means a fraction of strata (not rows) when
        using ``loss='profiled_poisson'``. Groups are shuffled each epoch while
        all rows belonging to a selected group stay in the same batch.
        """
        g = group_index
        n = int(g.numel())
        G = int(g.max().item()) + 1
        groups_per_batch = max(1, min(int(groups_per_batch), G))

        if groups_per_batch >= G:
            return [(
                torch.arange(n, device=g.device),
                g,
                G,
            )]

        counts = torch.bincount(g, minlength=G)
        group_perm = torch.randperm(G, device=g.device)

        # Rank each group according to this epoch's random group order, then
        # sort rows once so each batch is a contiguous slice of row_order.
        rank = torch.empty(G, device=g.device, dtype=torch.long)
        rank[group_perm] = torch.arange(G, device=g.device)
        row_order = torch.argsort(rank[g], stable=True)
        ordered_counts = counts[group_perm]
        cumulative_rows = torch.cat([
            torch.zeros(1, device=g.device, dtype=torch.long),
            torch.cumsum(ordered_counts, dim=0),
        ])

        out = []
        for start_g in range(0, G, groups_per_batch):
            stop_g = min(start_g + groups_per_batch, G)
            start_r = int(cumulative_rows[start_g].item())
            stop_r = int(cumulative_rows[stop_g].item())
            idx = row_order[start_r:stop_r]
            selected_counts = ordered_counts[start_g:stop_g]
            local_groups = torch.repeat_interleave(
                torch.arange(stop_g - start_g, device=g.device),
                selected_counts,
            )
            out.append((idx, local_groups, stop_g - start_g))
        return out

    def _batch_plan(self, n):
        if self.tcfg.batch_fraction is None:
            return n, 1
        bs = max(1, math.ceil(self.tcfg.batch_fraction * n))
        return bs, math.ceil(n / bs)

    @staticmethod
    def _batch(inputs, y, perm, s, bs, n):
        if perm is None:
            return inputs, y
        idx = perm[s * bs: min((s + 1) * bs, n)]
        return {k: v[idx] for k, v in inputs.items()}, y[idx]

    def _make_sched(self, opt, total_steps):
        if self.tcfg.schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(total_steps, 1), eta_min=self.tcfg.lr_min
            )
        return None

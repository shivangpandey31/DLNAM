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


def print_gpu_stats(prefix: str = "") -> None:
    """Print v1-style CUDA memory diagnostics when a GPU is active."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"{prefix}GPU allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB | "
        f"max allocated: {max_allocated:.2f} GB | "
        f"max reserved: {max_reserved:.2f} GB"
    )


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
        valid = {("log", "poisson"), ("logit", "bernoulli"), ("identity", "gaussian")}
        pair = (model_config.link, train_config.loss)
        if pair not in valid:
            allowed = ", ".join(f"{link}/{loss}" for link, loss in sorted(valid))
            raise ValueError(
                f"unsupported link/loss pair {pair[0]}/{pair[1]}; "
                f"supported pairs are {allowed}"
            )

    def fit(self, inputs: dict, y: torch.Tensor, *,
            validation_inputs: Optional[dict] = None,
            validation_y: Optional[torch.Tensor] = None):
        """Fit the vectorised ensemble.

        Supplying validation data changes only the early-stopping monitor; model
        updates still use ``inputs``/``y``. Validation inputs must already be on
        model scale (normally produced by the same fitted DataProcessor/scalers).
        """
        if (validation_inputs is None) != (validation_y is None):
            raise ValueError("validation_inputs and validation_y must be supplied together")

        t = self.tcfg
        start = time.perf_counter()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        y = y.to(self.device)
        if validation_inputs is not None:
            validation_inputs = {k: v.to(self.device) for k, v in validation_inputs.items()}
            validation_y = validation_y.to(self.device)

        if t.show_progress:
            bs, n_steps = self._batch_plan(int(y.shape[0]))
            print("\nDLNAM training")
            print(f"  device:       {self.device}")
            if self.device.type == "cuda":
                print(f"  GPU:          {torch.cuda.get_device_name(self.device)}")
            print(f"  samples:      {int(y.shape[0]):,}")
            print(f"  ensemble:     {len(self.ensemble)}")
            print(f"  batch size:   {bs:,} ({n_steps} steps/epoch)")
            print(f"  epochs:       {t.epochs}")
            if self.device.type == "cuda" and t.gpu_diagnostics:
                print_gpu_stats(prefix="  ")

        y_mean = float(y.mean())
        for m in self.ensemble:
            m.init_intercept(y_mean)
            m.train()

        run = self._fit_vectorised(
            inputs, y,
            validation_inputs=validation_inputs,
            validation_y=validation_y,
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
        }
        if t.show_progress:
            print(f"  elapsed:      {_format_seconds(elapsed)}")
        return self

    # ------------------------------------------------------------------
    # Vectorised ensemble: one vmap pass trains every member
    # ------------------------------------------------------------------
    def _fit_vectorised(self, inputs, y, *, validation_inputs=None, validation_y=None):
        t = self.tcfg
        n = y.shape[0]
        base = copy.deepcopy(self.ensemble[0]).to("meta")
        params, buffers = stack_module_state(self.ensemble)
        params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}

        def fmodel(p, b, inp):
            return functional_call(base, (p, b), (inp,), {"return_penalty": True})
        batched = vmap(fmodel, in_dims=(0, 0, None), randomness="different")

        strata_names = {
            name for name, spec in self.cfg.terms.items()
            if isinstance(spec, CategoricalTermSpec) and spec.role == "strata"
        }
        strata_prefixes = tuple(f"terms.{name}." for name in strata_names)
        regular_params, strata_params = [], []
        for key, value in params.items():
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
        if self.device.type == "cuda" and t.gpu_diagnostics:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(t.epochs + 1):
            epoch_start = time.perf_counter()
            perm = torch.randperm(n, device=self.device) if bs < n else None
            ep_sum = torch.zeros(len(self.ensemble), device=self.device)
            seen = 0

            for s in range(n_steps):
                bi, by = self._batch(inputs, y, perm, s, bs, n)
                opt.zero_grad(set_to_none=True)
                preds, penalties = batched(params, buffers, bi)
                per_m = _per_member_nll(preds, by, t.loss) + penalties
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

                    # Optional validation loss
                    monitor_text = ""
                    if validation_inputs is not None:
                        monitor_text = f" | Val {monitor:.4f}"

                    # Compact GPU diagnostics
                    gpu_text = ""
                    if self.device.type == "cuda" and t.gpu_diagnostics:
                        torch.cuda.synchronize(self.device)

                        allocated = (
                            torch.cuda.memory_allocated(self.device)
                            / 1024**3
                        )

                        peak = (
                            torch.cuda.max_memory_allocated(self.device)
                            / 1024**3
                        )

                        gpu_text = (
                            f" | GPU {allocated:.1f}G"
                            f"/{peak:.1f}G"
                        )

                    # Keep the entire status short enough to fit on one terminal line
                    msg = (
                        f"Epoch {epoch:4d}/{t.epochs} "
                        f"({progress * 100:5.1f}%)"
                        f" | Loss {float(ep_loss.mean()):.5f}"
                        f"{monitor_text}"
                        f" | LR {lr:.1e}"
                        f"{gpu_text}"
                        f" | {epoch_time:.1f}s"
                    )

                    # \r      -> return cursor to beginning of current line
                    # \033[K  -> clear existing text on that line
                    print(
                        "\r\033[K" + msg,
                        end="",
                        flush=True,
                    )

            if abort_training:
                break

        if t.show_progress:
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

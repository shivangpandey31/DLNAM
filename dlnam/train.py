"""
train.py - vectorised training for DLNAM ensembles.

All ensemble members train in one pass via torch.func
(stack_module_state + functional_call + vmap). Other engine features include
device-resident data, full-batch or permutation-indexed minibatch training,
link-matched loss, optional cosine scheduling, gradient clipping, a NaN guard,
and diagnostics decoupled from the optimisation step.

Scaling is handled by DataProcessor before training. The Trainer expects model
scale tensors and sets each member's intercept from the target mean as a warm
start.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Optional

import torch
from torch.func import stack_module_state, functional_call, vmap

from .config import ModelConfig, TrainConfig
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


def _per_member_nll(preds: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    """preds: (M, B, 1), y: (B, 1) -> (M,) per-member mean NLL."""
    yb = y.unsqueeze(0)
    if loss == "bernoulli":
        mu = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        elem = -yb * torch.log(mu) - (1.0 - yb) * torch.log(1.0 - mu)
    else:  # poisson
        elem = preds - yb * torch.log(preds + 1e-8)
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
        self.fit_summary: dict = {}

    @staticmethod
    def _validate_link_loss(model_config: ModelConfig, train_config: TrainConfig) -> None:
        valid = {("log", "poisson"), ("logit", "bernoulli")}
        pair = (model_config.link, train_config.loss)
        if pair not in valid:
            allowed = ", ".join(f"{link}/{loss}" for link, loss in sorted(valid))
            raise ValueError(
                f"unsupported link/loss pair {pair[0]}/{pair[1]}; "
                f"supported pairs are {allowed}"
            )

    # ------------------------------------------------------------------
    def fit(self, inputs: dict, y: torch.Tensor):
        t = self.tcfg
        start = time.perf_counter()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        y = y.to(self.device)

        y_mean = float(y.mean())
        for m in self.ensemble:
            m.init_intercept(y_mean)
            m.train()

        self._fit_vectorised(inputs, y)
        for m in self.ensemble:
            m.eval()
        elapsed = time.perf_counter() - start
        self.fit_summary = {
            "fit_seconds": float(elapsed),
            "epochs": int(t.epochs),
            "n_samples": int(y.shape[0]),
            "n_ensemble": int(len(self.ensemble)),
            "device": str(self.device),
        }
        if self._diagnostics_interval(t):
            print(f"  elapsed {_format_seconds(elapsed)}")
        return self

    # ------------------------------------------------------------------
    # Vectorised ensemble: one vmap pass trains every member
    # ------------------------------------------------------------------
    def _fit_vectorised(self, inputs, y):
        t = self.tcfg
        n = y.shape[0]
        base = copy.deepcopy(self.ensemble[0]).to("meta")
        params, buffers = stack_module_state(self.ensemble)
        params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}

        def fmodel(p, b, inp):
            return functional_call(base, (p, b), (inp,), {"return_penalty": True})
        batched = vmap(fmodel, in_dims=(0, 0, None), randomness="different")

        opt = torch.optim.AdamW(list(params.values()), lr=t.lr,
                                weight_decay=t.weight_decay)
        bs, n_steps = self._batch_plan(n)
        sched = self._make_sched(opt, t.epochs * n_steps)
        hist = [[] for _ in self.ensemble]
        diagnostics_every = self._diagnostics_interval(t)

        for epoch in range(t.epochs + 1):
            perm = torch.randperm(n, device=self.device) if bs < n else None
            # Sample-weighted running mean over the epoch, so a minibatch run
            # reports the epoch objective rather than a single batch: at
            # batch_fraction=0.01 the last batch is a 1% sample. Bookkeeping
            # only -- nothing here feeds the gradient.
            ep_sum, ep_seen = None, 0
            for s in range(n_steps):
                bi, by = self._batch(inputs, y, perm, s, bs, n)
                opt.zero_grad(set_to_none=True)
                preds, penalties = batched(params, buffers, bi)
                per_m = _per_member_nll(preds, by, t.loss) + penalties
                loss = per_m.mean()
                if torch.isnan(loss):
                    print(f"  warning: NaN objective at epoch {epoch}; stopping")
                    self._writeback(params, buffers); return
                loss.backward()
                if t.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params.values(), t.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                nb = int(by.shape[0])
                contrib = per_m.detach() * nb
                ep_sum = contrib if ep_sum is None else ep_sum + contrib
                ep_seen += nb
            ep_loss = ep_sum / max(ep_seen, 1)
            for i in range(len(self.ensemble)):
                hist[i].append((epoch, float(ep_loss[i])))
            if diagnostics_every and epoch and epoch % diagnostics_every == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  epoch {epoch:4d}/{t.epochs:<4d}   "
                      f"objective {float(ep_loss.mean()):9.4f}   lr {lr:.2e}")
        self._writeback(params, buffers)
        self.loss_history = hist

    def _writeback(self, params, buffers):
        for i, m in enumerate(self.ensemble):
            sd = {k: v[i].detach() for k, v in params.items()}
            sd.update({k: v[i].detach() for k, v in buffers.items()})
            missing, unexpected = m.load_state_dict(sd, strict=False)
            # strict=False ignores key mismatches, which would leave the member
            # holding its initial weights instead of the trained ones.
            if missing or unexpected:
                raise RuntimeError(
                    "vmap writeback key mismatch: trained weights not applied. "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}. "
                    "The stacked-state keys differ from the module state_dict; "
                    "the ensemble would keep initialisation values."
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
        s = self.tcfg.schedule
        if s == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(total_steps, 1), eta_min=self.tcfg.lr_min)
        return None

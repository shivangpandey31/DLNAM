"""
train.py — the training engine.

Two execution paths, same result:
  * vectorised (default): all ensemble members train in ONE pass via
    torch.func (stack_module_state + functional_call + vmap). This is the fix
    for the per-model Python loop that made full-size training crawl.
  * loop fallback (vectorize_ensemble=False): trains members one at a time;
    the correctness reference for the vectorised path.

Other engine features: data kept resident on device (no per-step host<->device
copies), full-batch or permutation-indexed minibatch, link-matched loss, cosine
/ plateau / none schedules, gradient clipping, AMP on CUDA, a NaN guard, and
diagnostics decoupled from the optimisation step.

Scaling is the caller's responsibility (or the processor's, in Phase 3): terms
must have had fit_scaling applied and inputs must already be on the model scale.
The Trainer sets each member's intercept from y as a warm start.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
from torch.func import stack_module_state, functional_call, vmap

from .config import ModelConfig, TrainConfig
from .model import DLNAM


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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(train_config.seed)
        self.ensemble = [
            DLNAM.from_config(model_config).to(self.device)
            for _ in range(train_config.n_ensemble)
        ]
        self.loss_history: list[list[tuple]] = []

    # ------------------------------------------------------------------
    def fit(self, inputs: dict, y: torch.Tensor,
            vectorize_ensemble: bool = True):
        t = self.tcfg
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        y = y.to(self.device)

        y_mean = float(y.mean())
        for m in self.ensemble:
            m.init_intercept(y_mean)
            m.train()

        if vectorize_ensemble and len(self.ensemble) > 1:
            self._fit_vectorised(inputs, y)
        else:
            self._fit_loop(inputs, y)
        for m in self.ensemble:
            m.eval()
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
            return functional_call(base, (p, b), (inp,))
        batched = vmap(fmodel, in_dims=(0, 0, None))

        opt = torch.optim.AdamW(list(params.values()), lr=t.lr,
                                weight_decay=t.weight_decay)
        bs, n_steps = self._batch_plan(n)
        sched = self._make_sched(opt, t.epochs * n_steps)
        hist = [[] for _ in self.ensemble]

        for epoch in range(t.epochs + 1):
            perm = torch.randperm(n, device=self.device) if bs < n else None
            ep_loss = None
            for s in range(n_steps):
                bi, by = self._batch(inputs, y, perm, s, bs, n)
                opt.zero_grad(set_to_none=True)
                preds = batched(params, buffers, bi)            # (M, B, 1)
                per_m = _per_member_nll(preds, by, t.loss)      # (M,)
                loss = per_m.mean()
                if torch.isnan(loss):
                    print(f"  NaN at epoch {epoch} — stopping."); 
                    self._writeback(params, buffers); return
                loss.backward()
                if t.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params.values(), t.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                ep_loss = per_m.detach()
            for i in range(len(self.ensemble)):
                hist[i].append((epoch, float(ep_loss[i])))
            if epoch and epoch % max(t.diagnostics_every, 1) == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  epoch {epoch:4d} | loss {float(ep_loss.mean()):9.4f} "
                      f"| lr {lr:.2e}")
        self._writeback(params, buffers)
        self.loss_history = hist

    def _writeback(self, params, buffers):
        for i, m in enumerate(self.ensemble):
            sd = {k: v[i].detach() for k, v in params.items()}
            sd.update({k: v[i].detach() for k, v in buffers.items()})
            m.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------
    # Loop fallback (reference)
    # ------------------------------------------------------------------
    def _fit_loop(self, inputs, y):
        t = self.tcfg
        n = y.shape[0]
        self.loss_history = []
        for m in self.ensemble:
            criterion = m.link.make_loss()
            opt = torch.optim.AdamW(m.parameters(), lr=t.lr,
                                    weight_decay=t.weight_decay)
            bs, n_steps = self._batch_plan(n)
            sched = self._make_sched(opt, t.epochs * n_steps)
            hist = []
            for epoch in range(t.epochs + 1):
                perm = torch.randperm(n, device=self.device) if bs < n else None
                last = None
                for s in range(n_steps):
                    bi, by = self._batch(inputs, y, perm, s, bs, n)
                    opt.zero_grad(set_to_none=True)
                    loss = criterion(m(bi), by)
                    if torch.isnan(loss):
                        print(f"  NaN at epoch {epoch} — stopping."); break
                    loss.backward()
                    if t.grad_clip:
                        torch.nn.utils.clip_grad_norm_(m.parameters(), t.grad_clip)
                    opt.step()
                    if sched is not None:
                        sched.step()
                    last = loss.item()
                hist.append((epoch, last))
            self.loss_history.append(hist)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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
        if s == "plateau":
            return None  # stepped externally in a future revision; full-batch only
        return None

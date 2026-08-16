"""
model.py — the model is just: intercept + sum of terms + link.

Inputs are passed as a dict keyed by term name, replacing the positional
(x_exposures, x_conf, x_time, x_encodings) signature. Adding a term type never
changes the forward signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .links import Link, make_link
from .terms.base import AdditiveTerm


class DLNAM(nn.Module):
    def __init__(self, terms: dict[str, AdditiveTerm], link: Link):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(1))
        self.terms = nn.ModuleDict(terms)
        self.link = link

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "DLNAM":
        """Build each term from its spec (dispatch on spec type), then assemble.
        Term construction order follows cfg.terms insertion order, so RNG state
        is deterministic for a given config + seed."""
        from .config import (SurfaceTermSpec, SmoothTermSpec,
                             TrendTermSpec, CategoricalTermSpec)
        from .terms.surface import SurfaceTerm
        from .terms.smooth import SmoothTerm, TrendTerm
        from .terms.categorical import CategoricalTerm

        builders = {
            SurfaceTermSpec:     SurfaceTerm,
            SmoothTermSpec:      SmoothTerm,
            TrendTermSpec:       TrendTerm,
            CategoricalTermSpec: CategoricalTerm,
        }
        terms: dict[str, AdditiveTerm] = {}
        for name, spec in cfg.terms.items():
            builder = builders.get(type(spec))
            if builder is None:
                raise NotImplementedError(
                    f"no builder for spec type {type(spec).__name__}"
                )
            terms[name] = builder(name, spec)
        return cls(terms, make_link(cfg.link))

    def forward(self, inputs: dict[str, torch.Tensor], *,
                return_penalty: bool = False,
                return_eta: bool = False):
        """Evaluate the additive model.

        ``return_eta=True`` exposes the linear predictor before the inverse
        link. The profiled-Poisson likelihood uses this scale to eliminate
        nuisance stratum intercepts analytically instead of estimating an ID
        embedding.
        """
        eta = self.intercept
        penalty = eta.new_zeros(())
        for name, term in self.terms.items():
            contribution = term(inputs[name])
            eta = eta + contribution
            weight = float(getattr(getattr(term, "spec", None), "penalty", 0.0))
            if weight > 0.0:
                penalty = penalty + weight * contribution.pow(2).mean()
        output = eta if return_eta else self.link.inverse(eta)
        if return_penalty:
            return output, penalty
        return output

    def term(self, name: str) -> AdditiveTerm:
        return self.terms[name]

    def init_intercept(self, y_mean: float) -> None:
        """Set the intercept from the target mean on the link scale. Exact only
        if every term starts near zero effect; the ExU surfaces do not, so treat
        this as a warm start rather than an exact mean-matching."""
        import math
        if self.link.name == "log":
            val = math.log(max(y_mean, 1e-8))
        elif self.link.name == "logit":
            p = min(max(y_mean, 1e-6), 1 - 1e-6)
            val = math.log(p / (1 - p))
        else:
            val = y_mean
        with torch.no_grad():
            self.intercept.fill_(val)

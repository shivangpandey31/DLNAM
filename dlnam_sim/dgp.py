"""
dlnam_sim/dgp.py — simulation add-on. Imports FROM dlnam only.

A "true" effect is a FunctionTerm implementing the same AdditiveTerm.effect()
contract as a learned term, so ground truth flows through the identical
extraction/centering path. DataGeneratingProcess composes true terms +
covariate generators + link + sampler into a reproducible simulator whose
output DataFrame is consumed by the SAME DataProcessor used on real data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from dlnam.links import Link, make_link
from dlnam.terms.base import AdditiveTerm, Centering, EffectCurve
from dlnam.data import make_windows


# ---------------------------------------------------------------------------
# Ground-truth term (analytic, not trained)
# ---------------------------------------------------------------------------

class FunctionTerm(AdditiveTerm):
    """Analytic ground-truth term, one per kind.

    kind='surface'     fn(value, lag) -> log-RR    (cumulated over lag)
    kind='smooth'      fn(value)      -> log effect
    kind='trend'       fn(time)       -> log effect, time in [0,1]
    kind='categorical' fn ignored; pass `effects` (C,) per-level log effects

    fn must be numpy/torch-polymorphic (pure arithmetic) for the continuous
    kinds, so the same function serves both data generation (torch) and the
    truth curve (numpy).
    """
    def __init__(self, name: str, fn: Optional[Callable] = None, *,
                 kind: str = "smooth", lag_max: int = 0,
                 value_range: tuple = (0.0, 1.0),
                 num_categories: Optional[int] = None,
                 order: Optional[list] = None,
                 effects: Optional[np.ndarray] = None):
        super().__init__(name)
        self.fn = fn
        self.kind = kind
        self.lag_max = lag_max
        self._value_range = value_range
        self.num_categories = num_categories
        self.order = order or ([] if num_categories is None
                               else [str(i) for i in range(num_categories)])
        self.cat_effects = None if effects is None else np.asarray(effects, float)
        if kind == "trend":
            self._value_range = (0.0, 1.0); self._data_median = 0.5
        elif kind == "categorical":
            self._data_median = float((num_categories or 1) // 2)
        else:
            self._data_median = float(np.mean(value_range))

    # --- data generation (eta) ------------------------------------------
    def forward(self, x):
        if self.kind == "surface":
            Lp1 = x.shape[1]
            lags = torch.arange(Lp1, dtype=x.dtype, device=x.device).unsqueeze(0)
            return self.fn(x, lags).sum(dim=1, keepdim=True)
        if self.kind == "categorical":
            eff = torch.tensor(self.cat_effects, dtype=torch.float32, device=x.device)
            return eff[x.long()].view(-1, 1)
        xx = x if x.ndim == 2 else x.view(-1, 1)
        return self.fn(xx).view(-1, 1)

    # --- truth curve ----------------------------------------------------
    def default_grid(self, n: int = 201) -> np.ndarray:
        if self.kind == "categorical":
            return np.arange(self.num_categories, dtype=float)
        lo, hi = self._value_range
        return np.linspace(lo, hi, n)

    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        g = np.asarray(grid_raw, dtype=float)
        if self.kind == "surface":
            lags = np.arange(self.lag_max + 1)[None, :]
            return np.asarray(self.fn(g[:, None], lags)).sum(axis=1)
        if self.kind == "categorical":
            return self.cat_effects[g.round().astype(int)]
        return np.asarray(self.fn(g))


# ---------------------------------------------------------------------------
# Outcome samplers
# ---------------------------------------------------------------------------

class Sampler(ABC):
    @abstractmethod
    def sample(self, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


class PoissonSampler(Sampler):
    def sample(self, mu, rng):
        return rng.poisson(np.clip(mu, 0, None))


class BernoulliSampler(Sampler):
    def sample(self, mu, rng):
        return rng.binomial(1, np.clip(mu, 0, 1))


class NegBinSampler(Sampler):
    """Overdispersed counts: Var = mu + mu^2/r. Lets you fit a Poisson DLNAM
    under true overdispersion — the misspecification stress test."""
    def __init__(self, r: float = 10.0):
        self.r = r

    def sample(self, mu, rng):
        mu = np.clip(mu, 1e-8, None)
        p = self.r / (self.r + mu)
        return rng.negative_binomial(self.r, p)


# ---------------------------------------------------------------------------
# The DGP
# ---------------------------------------------------------------------------

@dataclass
class SimulatedDataset:
    frame: pd.DataFrame
    n: int


@dataclass
class DataGeneratingProcess:
    true_terms: dict                  # name -> FunctionTerm
    covariate_sampler: dict           # name -> fn(T, rng) -> raw array (not trend)
    intercept: float = 0.0
    link: Link = field(default_factory=lambda: make_link("log"))
    sampler: Sampler = field(default_factory=PoissonSampler)
    target_col: str = "y"

    def _total_lag(self) -> int:
        lags = [t.lag_max for t in self.true_terms.values() if t.kind == "surface"]
        return max(lags) if lags else 0

    def simulate(self, n: int, seed: int) -> SimulatedDataset:
        rng = np.random.default_rng(seed)
        total_lag = self._total_lag()
        T = n + total_lag

        raw = {name: np.asarray(s(T, rng)) for name, s in self.covariate_sampler.items()}

        # raw inputs aligned to n samples, for computing eta via true terms
        eta = torch.full((n, 1), float(self.intercept))
        for name, term in self.true_terms.items():
            if term.kind == "surface":
                win = make_windows(raw[name], term.lag_max)[total_lag - term.lag_max:]
                xi = torch.tensor(win, dtype=torch.float32)
            elif term.kind == "trend":
                xi = torch.tensor(np.linspace(0, 1, n, dtype=np.float32)).view(-1, 1)
            elif term.kind == "categorical":
                xi = torch.tensor(raw[name][total_lag:].astype(np.int64))
            else:
                xi = torch.tensor(raw[name][total_lag:], dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                eta = eta + term.forward(xi)
        mu = self.link.inverse(eta).numpy().ravel()
        y = self.sampler.sample(mu, rng)

        # full-length DataFrame for DataProcessor (it windows + trims itself)
        cols = {}
        for name, term in self.true_terms.items():
            if term.kind == "trend":
                continue                                   # synthetic, no column
            if term.kind == "categorical":
                cols[name] = [term.order[i] for i in raw[name].astype(int)]
            else:
                cols[name] = raw[name]
        ytarget = np.zeros(T); ytarget[total_lag:] = y
        cols[self.target_col] = ytarget
        return SimulatedDataset(frame=pd.DataFrame(cols), n=n)

    def truth_curve(self, name: str, grid_raw, centering: Centering) -> EffectCurve:
        return self.true_terms[name].effect(grid_raw, centering)

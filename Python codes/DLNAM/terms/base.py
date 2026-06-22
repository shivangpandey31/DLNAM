"""
terms/base.py — the AdditiveTerm interface. THIS IS THE CENTRAL ABSTRACTION.

Every additive component implements it: exposure surfaces, smooth confounders,
the trend, categoricals — AND the ground-truth functions used in simulation.
Because a learned term and a true term expose the *same* `effect()` contract,
bias and coverage are simply pointwise comparisons between two objects of the
same type, and the evaluator/visualizer never need to know which is which.
(That symmetry is what lets the simulation package be a pure add-on: it
implements this interface and reuses everything else unchanged.)

The two halves of the contract:
  forward(x)  -> (B, 1)   the term's contribution to the linear predictor eta.
                          Used during training and prediction.
  effect(...) -> EffectCurve  the centered additive effect on an interpretable
                          grid, in *raw* input units. Used for plotting,
                          for comparison against truth, and for uncertainty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Centering — one explicit definition, used everywhere
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Centering:
    """How an additive effect is anchored so it reads 0 (RR/OR = 1) somewhere.

    method:
      'reference' — subtract the effect at a specific raw input value (the
                    DLNM cenvalue convention). `value` required.
      'median'    — subtract the effect at the data median (NOT, as in the old
                    code, the middle of an arbitrary evaluation grid).
      'mean'      — subtract the mean effect over the training inputs.
      'custom'    — alias of 'reference' with an explicit value.
    """
    method: Literal["reference", "median", "mean", "custom"] = "reference"
    value: Optional[float] = None

    def anchor(self,
               grid_raw: np.ndarray,
               log_effect_on_grid: np.ndarray,
               train_log_effect: Optional[np.ndarray] = None,
               data_median: Optional[float] = None) -> float:
        """Return the scalar to subtract from `log_effect_on_grid`."""
        if self.method == "mean":
            if train_log_effect is None:
                raise ValueError("centering='mean' needs train_log_effect")
            return float(np.mean(train_log_effect))

        if self.method == "median":
            if data_median is None:
                raise ValueError("centering='median' needs data_median")
            ref = data_median
        elif self.method in ("reference", "custom"):
            if self.value is None:
                raise ValueError(f"centering='{self.method}' needs value")
            ref = self.value
        else:
            raise ValueError(f"unknown centering method: {self.method}")

        idx = int(np.argmin(np.abs(np.asarray(grid_raw) - ref)))
        return float(log_effect_on_grid[idx])


@dataclass
class EffectCurve:
    """The interpretable output of a term. The shared currency between learned
    terms and ground truth (truth simply leaves the uncertainty fields None).

    grid_raw   : (G,) evaluation points in raw input units (x-axis)
    log_effect : (G,) centered additive effect (log-RR / logit space)
    per_lag    : (n_lags, G) optional, surfaces only
    label      : x-axis label
    """
    name: str
    grid_raw: np.ndarray
    log_effect: np.ndarray
    per_lag: Optional[np.ndarray] = None
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

class AdditiveTerm(nn.Module, ABC):
    """Common interface for everything that adds into the linear predictor."""

    def __init__(self, name: str, scaling: str = "minmax"):
        super().__init__()
        self.name = name
        self.scaling = scaling
        # Set by fit_scaling on data-aware terms; used for 'median' centering
        # and for deriving default grids. None until fitted.
        self._data_median: Optional[float] = None
        self.label: Optional[str] = None

    # --- training / prediction half -------------------------------------
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is this term's slice of the model input. Returns (B, 1)."""

    # --- interpretation / validation half -------------------------------
    @abstractmethod
    def default_grid(self) -> np.ndarray:
        """A natural evaluation grid in RAW input units, derived from this
        term's own scaling/data range (so nothing downstream hardcodes a
        range like linspace(-2, 2) that silently assumes z-scoring)."""

    @abstractmethod
    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        """Uncentered additive effect on `grid_raw`. For surfaces this is the
        cumulative-over-lag log-RR. Centering is applied by `effect()`."""

    def effect(self,
               grid_raw: Optional[np.ndarray] = None,
               centering: Optional[Centering] = None) -> EffectCurve:
        """Template method: evaluate raw effect, then center. Concrete terms
        override `raw_log_effect`/`default_grid`, not this."""
        if grid_raw is None:
            grid_raw = self.default_grid()
        grid_raw = np.asarray(grid_raw, dtype=float)
        if centering is None:
            centering = Centering(method="median")

        log = np.asarray(self.raw_log_effect(grid_raw), dtype=float)
        anchor = centering.anchor(grid_raw, log, data_median=self._data_median)
        return EffectCurve(
            name=self.name,
            grid_raw=grid_raw,
            log_effect=log - anchor,
            label=self.label or self.name,
        )

    # --- introspection used by inference/penalties ----------------------
    def parameters_l2(self) -> torch.Tensor:
        """Optional roughness/output penalty contribution; default 0."""
        return torch.zeros(())

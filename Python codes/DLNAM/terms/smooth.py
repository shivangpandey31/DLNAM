"""
terms/smooth.py — 1-D additive terms: continuous confounders and the trend.

Both are the same shape (scalar in, scalar contribution out, subnet-mixed), so
they share `_OneDTerm`. They differ only in how the raw input maps to the model
scale and in their natural evaluation grid:

  SmoothTerm — a confounder; scaled by fit_scaling, grid over the data range,
               centred at the data median.
  TrendTerm  — normalised time in [0, 1]; no scaling, grid over [0, 1],
               centred at the midpoint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import SmoothTermSpec, TrendTermSpec, ExUSpec
from ..layers import ExULayer, init_linear, init_mix
from .base import AdditiveTerm


def _build_1d_subnet(spec, input_exu: Optional[ExUSpec]) -> nn.Sequential:
    layers = list(spec.layers)
    if not layers:                                     # degenerate: scalar->scalar
        return nn.Sequential(nn.Linear(1, 1))
    first_w = layers[0].width
    mods = []
    if input_exu is not None and input_exu.enabled:
        mods.append(ExULayer(1, first_w, input_exu.weight_mean, input_exu.weight_std,
                             input_exu.bias_init))
    else:
        mods.append(init_linear(nn.Linear(1, first_w),
                                layers[0].weight_init, layers[0].bias_init))
    mods.append(layers[0].activation.build())
    last = first_w
    for ls in layers[1:]:
        mods.append(init_linear(nn.Linear(last, ls.width), ls.weight_init, ls.bias_init))
        mods.append(ls.activation.build())
        if ls.dropout > 0:
            mods.append(nn.Dropout(ls.dropout))
        last = ls.width
    mods.append(nn.Linear(last, 1))
    return nn.Sequential(*mods)


class _OneDTerm(AdditiveTerm):
    """Shared implementation for 1-D smooth terms."""

    def __init__(self, name: str, spec, input_exu: Optional[ExUSpec]):
        super().__init__(name, scaling=spec.scaling)
        self.spec = spec
        self.subnets = nn.ModuleList(
            [_build_1d_subnet(spec, input_exu) for _ in range(spec.num_subnets)]
        )
        self.mix = nn.Parameter(init_mix(spec))
        self._loc = 0.0
        self._scale = 1.0
        self._value_range = (0.0, 1.0)

    # --- scaling ---------------------------------------------------------
    def fit_scaling(self, raw_values: np.ndarray) -> None:
        raw = np.asarray(raw_values, dtype=float)
        if self.scaling == "zscore":
            self._loc, self._scale = float(raw.mean()), float(raw.std() + 1e-12)
        elif self.scaling == "none":
            self._loc, self._scale = 0.0, 1.0
        else:  # minmax
            self._loc, self._scale = float(raw.min()), float(raw.max() - raw.min() + 1e-12)
        self._value_range = (float(raw.min()), float(raw.max()))
        self._data_median = float(np.median(raw))

    def _to_scaled(self, raw: np.ndarray) -> np.ndarray:
        if self.scaling == "none":
            return np.asarray(raw, dtype=float)
        return (np.asarray(raw, dtype=float) - self._loc) / self._scale

    # --- mixing ----------------------------------------------------------
    def _mix(self):
        return torch.softmax(self.mix, dim=0) if self.spec.constrain_subnet_weights else self.mix

    def _mixed(self, x: torch.Tensor) -> torch.Tensor:
        w = self._mix()
        outs = torch.stack([s(x) for s in self.subnets], dim=0)   # (S, N, 1)
        return (w.view(-1, 1, 1) * outs).sum(dim=0)               # (N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mixed(x)                                     # x: (B, 1) scaled

    def default_grid(self, n: int = 200) -> np.ndarray:
        lo, hi = self._value_range
        return np.linspace(lo, hi, n)

    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        gs = self._to_scaled(grid_raw)
        device = self.mix.device
        x = torch.tensor(gs, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            return self._mixed(x).squeeze(1).cpu().numpy()


class SmoothTerm(_OneDTerm):
    """1-D smooth function of a continuous confounder."""
    def __init__(self, name: str, spec: SmoothTermSpec):
        super().__init__(name, spec, input_exu=spec.input_exu)


class TrendTerm(_OneDTerm):
    """1-D long-term trend over normalised time in [0, 1]. Input is already
    [0, 1] by construction, so scaling is identity and the grid is [0, 1]."""
    def __init__(self, name: str, spec: TrendTermSpec):
        super().__init__(name, spec, input_exu=spec.input_exu)
        self._value_range = (0.0, 1.0)
        self._data_median = 0.5

    def fit_scaling(self, raw_values=None) -> None:
        # Trend needs no data-driven scaling; keep identity + [0,1] grid.
        self._loc, self._scale = 0.0, 1.0
        self._value_range = (0.0, 1.0)
        self._data_median = 0.5

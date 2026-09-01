"""
terms/surface.py — the exposure value x lag surface, plus the pluggable
multivariate-ExU encoder strategies.

The encoder is the *only* place the 2-D (value, lag) ExU scheme lives, so
encoder comparisons can be made through a term-spec change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from ..config import SurfaceTermSpec
from ..layers import (ExULayer, ExULayerSharedBias, ExULayerLocalBias,
                      apply_subnet_dropout, init_linear, init_mix)
from .base import AdditiveTerm


# ---------------------------------------------------------------------------
# Surface input-encoder strategies
# ---------------------------------------------------------------------------

class SurfaceEncoder(nn.Module, ABC):
    """Maps a value column and a lag column (each (N, 1)) to (N, out_features)."""

    @abstractmethod
    def forward(self, value: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        ...


class LinearEncoder(SurfaceEncoder):
    """No-ExU ablation: a plain linear input layer over (value, lag), i.e. a
    standard NAM/MLP first layer with no exponential-of-weight parameterisation
    and no per-dimension ExU localisation. This is the 'regular DLNAM' baseline."""

    def __init__(self, out_features: int, activation_factory):
        super().__init__()
        self.lin = nn.Linear(2, out_features)
        self.act = activation_factory()

    def forward(self, value, lag):
        return self.act(self.lin(torch.cat([value, lag], dim=1)))


class SplitConcatEncoder(SurfaceEncoder):
    """Separate value and lag encoders whose features are concatenated."""

    def __init__(self, out_features: int, activation_factory,
                 weight_mean_val: float, weight_mean_lag: float,
                 weight_std: float, use_exu_lag: bool, bias_init=None):
        super().__init__()
        half = out_features // 2
        rem = out_features - half
        self.act = activation_factory()
        self.exu_val = ExULayer(1, rem, weight_mean_val, weight_std, bias_init)
        self.use_exu_lag = use_exu_lag
        if use_exu_lag:
            self.lag = ExULayer(1, half, weight_mean_lag, weight_std, bias_init)
        else:
            self.lag = nn.Linear(1, half)

    def forward(self, value, lag):
        vf = self.act(self.exu_val(value))
        if self.use_exu_lag:
            lf = self.act(self.lag(lag))
        else:
            lf = torch.sigmoid(self.lag(lag))
        return torch.cat([vf, lf], dim=1)


class Joint2DExUEncoder(SurfaceEncoder):
    """Unified ExU with one bias per input dimension.
    h = sigma(e^W (x - b)). Mixes value/lag in the input layer; no per-unit
    localisation."""

    def __init__(self, out_features: int, activation_factory,
                 weight_mean_val: float, weight_mean_lag: float, weight_std: float,
                 bias_init=None):
        super().__init__()
        self.act = activation_factory()
        self.exu = ExULayerSharedBias(2, out_features,
                                      [weight_mean_val, weight_mean_lag], weight_std,
                                      bias_init)

    def forward(self, value, lag):
        return self.act(self.exu(torch.cat([value, lag], dim=1)))


class UnifiedLocalBiasEncoder(SurfaceEncoder):
    """Unified ExU with a bias per unit and per input dimension.
    h = sigma((e^W ⊙ (1 x^T - B)) 1). Mixes value/lag in the input layer while
    preserving per-unit localisation; reduces to scalar ExU at k0=1."""

    def __init__(self, out_features: int, activation_factory,
                 weight_mean_val: float, weight_mean_lag: float, weight_std: float,
                 bias_init=None):
        super().__init__()
        self.act = activation_factory()
        self.exu = ExULayerLocalBias(2, out_features,
                                     [weight_mean_val, weight_mean_lag], weight_std,
                                     bias_init)

    def forward(self, value, lag):
        return self.act(self.exu(torch.cat([value, lag], dim=1)))


def make_surface_encoder(spec: SurfaceTermSpec) -> SurfaceEncoder:
    out_features = spec.layers[0].width
    act = spec.layers[0].activation.build      # honor transform (e.g. softcap) on input layer
    exu = spec.input_exu
    strategy = exu.surface_strategy if exu is not None else "concat"
    mean_val = exu.weight_mean if exu else 1.5
    mean_lag = (exu.weight_mean_lag if exu and exu.weight_mean_lag is not None
                else mean_val)
    std = exu.weight_std if exu else 0.5
    bias_init = exu.bias_init if exu else None

    # No ExU at all -> plain linear input layer (true no-ExU ablation).
    if exu is None or not exu.enabled:
        return LinearEncoder(out_features=out_features, activation_factory=act)

    if strategy == "concat":
        return SplitConcatEncoder(
            out_features=out_features, activation_factory=act,
            weight_mean_val=mean_val, weight_mean_lag=mean_lag,
            weight_std=std, use_exu_lag=True, bias_init=bias_init)
    if strategy == "unified_shared_bias":
        return Joint2DExUEncoder(out_features, act, mean_val, mean_lag, std, bias_init)
    if strategy == "unified_local_bias":
        return UnifiedLocalBiasEncoder(out_features, act, mean_val, mean_lag, std, bias_init)
    raise ValueError(f"unknown surface strategy '{strategy}'")


# ---------------------------------------------------------------------------
# One subnet (encoder -> hidden stack -> scalar)
# ---------------------------------------------------------------------------

class _SurfaceSubnet(nn.Module):
    def __init__(self, spec: SurfaceTermSpec):
        super().__init__()
        self.encoder = make_surface_encoder(spec)
        self.encoder_dropout = (nn.Dropout(spec.layers[0].dropout)
                                if spec.layers[0].dropout > 0 else nn.Identity())
        last = spec.layers[0].width
        tail = []
        for ls in spec.layers[1:]:
            tail.append(init_linear(nn.Linear(last, ls.width),
                                    ls.weight_init, ls.bias_init))
            tail.append(ls.activation.build())
            if ls.dropout > 0:
                tail.append(nn.Dropout(ls.dropout))
            last = ls.width
        tail.append(nn.Linear(last, 1))
        self.tail = nn.Sequential(*tail)

    def forward(self, vl: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_dropout(self.encoder(vl[:, 0:1], vl[:, 1:2]))
        return self.tail(feat)


# ---------------------------------------------------------------------------
# The term
# ---------------------------------------------------------------------------

class SurfaceTerm(AdditiveTerm):
    """value x lag distributed-lag surface.

    Holds `num_subnets` subnets, mixed by learned weights; forward() sums the
    per-lag contributions over the lag window.
    """

    def __init__(self, name: str, spec: SurfaceTermSpec):
        super().__init__(name, scaling=spec.scaling)
        self.spec = spec
        self.lag_max = spec.lag_max
        self.subnets = nn.ModuleList(
            [_SurfaceSubnet(spec) for _ in range(spec.num_subnets)]
        )
        self.mix = nn.Parameter(init_mix(spec))
        self.register_buffer("lag_grid", torch.linspace(0.0, 1.0, spec.lag_max + 1))
        # Fixed grid in scaled coordinates for the curvature penalty. Registered
        # as a buffer so it moves with the module and is carried by the stacked
        # ensemble state under torch.func; it holds no parameters.
        self._penalise = (float(spec.roughness_value) > 0.0
                          or float(spec.roughness_lag) > 0.0)
        if self._penalise:
            gv = torch.linspace(0.0, 1.0, int(spec.roughness_grid))
            vv = gv.repeat_interleave(spec.lag_max + 1).unsqueeze(1)
            ll = torch.linspace(0.0, 1.0, spec.lag_max + 1).repeat(
                int(spec.roughness_grid)).unsqueeze(1)
            self.register_buffer("rough_grid", torch.cat([vv, ll], dim=1))
        # scaling params, set by fit_scaling
        self._loc = 0.0
        self._scale = 1.0
        self._value_range = (0.0, 1.0)

    # --- scaling (set from data once, outside the optimiser) -------------
    def fit_scaling(self, raw_values: np.ndarray) -> None:
        raw = np.asarray(raw_values, dtype=float)
        if self.scaling == "zscore":
            self._loc = float(raw.mean())
            self._scale = float(raw.std() + 1e-12)
        else:  # minmax -> [0, 1]
            self._loc = float(raw.min())
            self._scale = float(raw.max() - raw.min() + 1e-12)
        self._value_range = (float(raw.min()), float(raw.max()))
        self._data_median = float(np.median(raw))

    def _to_scaled(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=float) - self._loc) / self._scale

    # --- mixing ----------------------------------------------------------
    def _mix(self) -> torch.Tensor:
        if self.spec.constrain_subnet_weights:
            w = torch.softmax(self.mix, dim=0)
        else:
            w = self.mix
        return apply_subnet_dropout(w, self.spec.subnet_dropout, self.training)

    def _mixed(self, vl: torch.Tensor) -> torch.Tensor:
        w = self._mix()
        outs = torch.stack([s(vl) for s in self.subnets], dim=0)   # (S, N, 1)
        return (w.view(-1, 1, 1) * outs).sum(dim=0)                 # (N, 1)

    # --- training / prediction ------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lag_max + 1) scaled exposure window (col j = lag j)
        B, Lp1 = x.shape
        v = x.reshape(-1, 1)
        l = self.lag_grid.to(x.device).repeat(B).unsqueeze(1)
        vl = torch.cat([v, l], dim=1)                              # (B*Lp1, 2)
        mixed = self._mixed(vl)                                    # (B*Lp1, 1)
        return mixed.view(B, Lp1).sum(dim=1, keepdim=True)         # (B, 1)

    def roughness_penalty(self) -> torch.Tensor:
        """Mean squared second difference of the fitted surface, per margin.

        Evaluated on the fixed scaled grid, so it constrains the function the
        model reports rather than its parameters -- the same object a P-spline
        difference penalty acts on. Returns a scalar; zero when both weights
        are off.
        """
        if not self._penalise:
            return self.mix.new_zeros(())
        G = int(self.spec.roughness_grid)
        Lp1 = int(self.lag_max) + 1
        surface = self._mixed(self.rough_grid).view(G, Lp1)   # (value, lag)
        out = self.mix.new_zeros(())
        wv = float(self.spec.roughness_value)
        wl = float(self.spec.roughness_lag)
        # Divide by the squared grid spacing so each term approximates the mean
        # squared second derivative rather than a raw difference. Without this
        # the weight is expressed in units of "squared difference on this grid"
        # and changes meaning with the resolution, and between the two margins,
        # whose spacings differ (1/(G-1) against 1/lag_max).
        hv2 = (1.0 / max(G - 1, 1)) ** 2
        hl2 = (1.0 / max(self.lag_max, 1)) ** 2
        if wv > 0.0 and G >= 3:
            out = out + wv * (torch.diff(surface, n=2, dim=0) / hv2).pow(2).mean()
        if wl > 0.0 and Lp1 >= 3:
            out = out + wl * (torch.diff(surface, n=2, dim=1) / hl2).pow(2).mean()
        return out

    def _last_layer_design(self, x: torch.Tensor) -> torch.Tensor:
        """Exact design for the fixed-representation final linear layers."""
        B, Lp1 = x.shape
        v = x.reshape(-1, 1)
        l = self.lag_grid.to(x.device).repeat(B).unsqueeze(1)
        vl = torch.cat([v, l], dim=1)
        point_design = self._last_layer_point_design(vl)
        return point_design.view(B, Lp1, -1).sum(dim=1)

    def _last_layer_point_design(self, vl: torch.Tensor) -> torch.Tensor:
        """Exact final-layer design for individual exposure-lag points."""
        mix = self._mix()
        blocks = []
        for weight, subnet in zip(mix, self.subnets):
            feat = subnet.encoder_dropout(
                subnet.encoder(vl[:, 0:1], vl[:, 1:2])
            )
            feat = subnet.tail[:-1](feat)
            blocks.extend([
                weight * feat,
                weight.expand(vl.shape[0], 1),
            ])
        return torch.cat(blocks, dim=1)

    # --- interpretation --------------------------------------------------
    # An odd count puts the midpoint of a symmetric range on a grid node, which
    # matters because Centering.anchor snaps a reference value to the nearest
    # node: on-grid, reference centring is exact and agrees with the convention
    # the comparator software uses.
    def default_grid(self, n: int = 201) -> np.ndarray:
        lo, hi = self._value_range
        return np.linspace(lo, hi, n)

    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        """Cumulative-over-lag log-RR for a constant exposure held at each raw
        value across all lags. Returns (G,)."""
        gs = self._to_scaled(grid_raw)
        G = len(gs)
        lags = self.lag_grid.detach().cpu().numpy()
        Lp1 = len(lags)
        device = self.lag_grid.device
        v = torch.tensor(gs, dtype=torch.float32, device=device) \
            .view(-1, 1).repeat_interleave(Lp1, dim=0)            # value blocks
        l = torch.tensor(np.tile(lags, G), dtype=torch.float32, device=device) \
            .view(-1, 1)
        vl = torch.cat([v, l], dim=1)                             # (G*Lp1, 2)
        with torch.no_grad():
            cum = self._mixed(vl).view(G, Lp1).sum(dim=1)         # (G,)
        return cum.cpu().numpy()

    def per_lag_log_rr(self, grid_raw: np.ndarray, ref_raw: float,
                       lag_scaled: np.ndarray | None = None) -> np.ndarray:
        """Per-lag log-RR relative to a reference value: returns (n_lags, G),
        row k = log f(value, lag_k) - log f(ref, lag_k). Used for the lag
        surface (contour / 3-D) plot.

        `lag_scaled` overrides the observed lag grid with an arbitrary set of
        scaled lags in [0, 1]. The component takes the lag as a continuous
        network input, so evaluating between observed lags is a genuine model
        evaluation rather than interpolation of a coarse surface -- which is
        what a plot needs when the window has few lags.
        """
        gs = self._to_scaled(grid_raw)
        rs = float(self._to_scaled(np.asarray([ref_raw]))[0])
        G = len(gs)
        lags = (self.lag_grid.detach().cpu().numpy() if lag_scaled is None
                else np.asarray(lag_scaled, dtype=float))
        Lp1 = len(lags)
        device = self.lag_grid.device
        vv = torch.tensor(np.tile(gs, Lp1), dtype=torch.float32, device=device).view(-1, 1)
        rrv = torch.full((Lp1 * G, 1), rs, dtype=torch.float32, device=device)
        ll = torch.tensor(np.repeat(lags, G), dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            ev = self._mixed(torch.cat([vv, ll], dim=1)).view(Lp1, G)
            er = self._mixed(torch.cat([rrv, ll], dim=1)).view(Lp1, G)
        return (ev - er).cpu().numpy()

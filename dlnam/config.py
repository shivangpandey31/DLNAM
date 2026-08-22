"""
config.py — the typed specification system.

This is the single source of truth for *how a model is shaped*. Nothing here
imports torch tensors or builds modules; specs are plain, frozen, hashable,
serializable dataclasses. The model layer (`model.py`, `terms/`) consumes them.

Design goals this file is meant to satisfy:
  * Every term, and every *layer within a term*, is independently adjustable.
    There is no `[surface_layers] * n_exposures` broadcast baked into the model;
    broadcasting is an explicit, opt-in convenience (see `broadcast_terms`).
  * Activations are per-layer, and may carry an optional output *transform*
    (e.g. a learnable soft-cap) — see `ActivationSpec` / `TransformSpec`.
  * The multivariate ExU scheme for surfaces is a named strategy, not a hardcode.
  * Specs are explicit dataclasses so experiment configurations can be recorded
    alongside results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation + output transforms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformSpec:
    """Base class for an output transform applied *after* an activation.

    A transform is a small nn.Module mapping R -> R (elementwise). The learnable
    soft-cap is the motivating case but the slot is generic. `.build()` returns
    the module; keep transforms individually unit-testable (identity/limit
    behaviour should be obvious).
    """
    def build(self) -> nn.Module:        # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class SoftCapSpec(TransformSpec):
    """Smoothly bounds outputs to (-cap, +cap) via `cap * tanh(x / cap)`.

    cap is reparameterised through softplus so it stays positive; `learnable`
    decides whether it is a trained parameter or a fixed buffer.
    """
    init_cap: float = 8.0
    learnable: bool = True

    def build(self) -> nn.Module:
        from .activations import build_transform
        return build_transform(self)


# A factory returning a fresh activation module (e.g. `nn.Mish`). We store the
# *class/callable*, not an instance, so each layer gets its own module.
ActivationFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ActivationSpec:
    """An activation, optionally composed with an output transform.

    build() -> nn.Sequential(base(), transform.build()?) so it drops into any
    layer position uniformly.
    """
    base: ActivationFactory = nn.Mish
    transform: Optional[TransformSpec] = None

    def build(self) -> nn.Module:
        from .activations import build_activation
        return build_activation(self)


# ---------------------------------------------------------------------------
# ExU
# ---------------------------------------------------------------------------

SurfaceEncoderStrategy = Literal[
    "concat",
    "unified_shared_bias",
    "unified_local_bias",
]


@dataclass(frozen=True)
class ExUSpec:
    """ExU configuration for a term's input layer.

    weight_mean      mean of the log-domain weight init. Larger values give
                     sharper initial features.
    weight_mean_lag  surface terms only: separate mean for the LAG axis ExU
                     None -> reuse weight_mean.
    weight_std       std of the log-domain weight init.
    enabled          False -> linear input layer instead of ExU (and, for
                     surfaces, a linear+sigmoid lag encoder).
    surface_strategy how ExU generalises to the 2-D (value, lag) surface input.
    """
    enabled: bool = True
    weight_mean: float = 1.5
    weight_mean_lag: Optional[float] = None
    weight_std: float = 0.5
    surface_strategy: SurfaceEncoderStrategy = "concat"
    bias_init: Optional["InitSpec"] = None   # None -> uniform(0,1) (ExU default)


# ---------------------------------------------------------------------------
# Parameter initialisation (augmentable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InitSpec:
    """Initialisation scheme for a weight/bias/mixing tensor. Defaults reproduce
    a small-normal init; attach to LayerSpec (weight_init/bias_init), ExUSpec
    (bias_init), or TermSpec (mix_init)."""
    scheme: str = "normal"   # normal|uniform|constant|zeros|ones|xavier_uniform|
                             # xavier_normal|kaiming_uniform|kaiming_normal|
                             # torch_linear|orthogonal
    mean: float = 0.0
    std: float = 0.01
    lo: float = 0.0
    hi: float = 1.0
    value: float = 0.0
    gain: float = 1.4142135623730951   # sqrt(2), Kaiming/ReLU gain; used by orthogonal

    def apply_(self, t):
        s = self.scheme
        if s == "normal":              nn.init.normal_(t, self.mean, self.std)
        elif s == "uniform":           nn.init.uniform_(t, self.lo, self.hi)
        elif s == "constant":          nn.init.constant_(t, self.value)
        elif s == "zeros":             nn.init.zeros_(t)
        elif s == "ones":              nn.init.ones_(t)
        elif s == "xavier_uniform":    nn.init.xavier_uniform_(t)
        elif s == "xavier_normal":     nn.init.xavier_normal_(t)
        elif s == "kaiming_uniform":   nn.init.kaiming_uniform_(t, nonlinearity="relu")
        elif s == "kaiming_normal":    nn.init.kaiming_normal_(t, nonlinearity="relu")
        elif s == "orthogonal":
            # Orthogonal weight init (preserves gradient norms; lowers seed-to-seed
            # variance). Only defined for 2-D+ tensors; for a 1-D bias fall back to
            # a small normal. Apply to hidden/output LINEAR layers only -- NOT the
            # ExU input layer (its weights are exponentiated / live in log-space).
            if t.dim() >= 2:
                nn.init.orthogonal_(t, gain=self.gain)
            else:
                nn.init.normal_(t, self.mean, self.std)
        elif s == "torch_linear":      # exact PyTorch nn.Linear weight default
            import math
            if t.dim() >= 2:
                nn.init.kaiming_uniform_(t, a=math.sqrt(5))
            # 1-D (bias) torch_linear needs fan_in -> handled in init_linear()
        else: raise ValueError(f"unknown init scheme '{s}'")
        return t


# ---------------------------------------------------------------------------
# Layers and terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerSpec:
    """One hidden layer. width is the *output* width; input width is inferred
    by the builder from the previous layer / term input dimension."""
    width: int
    activation: ActivationSpec = field(default_factory=ActivationSpec)
    exu: Optional[ExUSpec] = None          # None -> plain nn.Linear
    dropout: float = 0.0
    weight_init: Optional[InitSpec] = None  # None -> PyTorch default
    bias_init: Optional[InitSpec] = None

    def __post_init__(self):
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("LayerSpec.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TermSpec:
    """Base spec shared by every additive term.

    layers       : the hidden stack; the final scalar-output layer is added by
                   the builder, so `layers` describes only the interior.
    num_subnets  : within-term subnet count (mixed by learned weights).
    scaling      : how this term's input is scaled ('minmax' | 'zscore' | 'none').
                   Travels WITH the term so downstream eval grids derive from it
                   instead of assuming a hardcoded range.
    penalty      : L2 penalty on the term contribution during training (0 = off).
                   Weight decay still controls parameter shrinkage globally.
    subnet_dropout : dropout probability applied to subnet contributions during
                   training. This is the subnetwork dropout regulariser used in
                   neural additive models; 0 disables it.
    constrain_subnet_weights : if True, the subnet mixing weights are softmaxed
                   into an exact CONVEX combination (sum to 1, non-negative);
                   if False (default) they are unconstrained real weights.
    mix_init     : initialisation for the subnet mixing weights. None ->
                   constant 1/num_subnets, which makes the contributions sum to
                   ~1 at init (raw) or exactly 1/S each (convex), aligning with a
                   single-subnet DLNAM at S=1.
    """
    layers: Sequence[LayerSpec] = ()
    num_subnets: int = 3
    scaling: Literal["minmax", "zscore", "none"] = "minmax"
    penalty: float = 0.0
    subnet_dropout: float = 0.0
    constrain_subnet_weights: bool = False
    mix_init: Optional[InitSpec] = None

    def __post_init__(self):
        if float(self.penalty) < 0.0:
            raise ValueError("TermSpec.penalty must be non-negative")
        if not 0.0 <= float(self.subnet_dropout) < 1.0:
            raise ValueError("TermSpec.subnet_dropout must be in [0, 1)")


@dataclass(frozen=True)
class SurfaceTermSpec(TermSpec):
    """Exposure value x lag surface (the DLNM piece)."""
    lag_max: int = 30
    # ExU for the value/lag input layer; surface_strategy lives inside it.
    input_exu: Optional[ExUSpec] = field(default_factory=ExUSpec)


@dataclass(frozen=True)
class SmoothTermSpec(TermSpec):
    """1-D smooth term for a continuous confounder."""
    scaling: Literal["minmax", "zscore", "none"] = "zscore"
    input_exu: Optional[ExUSpec] = None        # None -> linear first layer


@dataclass(frozen=True)
class TrendTermSpec(TermSpec):
    """1-D long-term trend over normalised time in [0, 1]."""
    scaling: Literal["minmax", "zscore", "none"] = "none"
    input_exu: Optional[ExUSpec] = field(default_factory=ExUSpec)


@dataclass(frozen=True)
class CategoricalTermSpec(TermSpec):
    """Categorical term. hidden==() -> pure embedding (lookup table)."""
    num_categories: int = 2
    order: Sequence[str] = ()              # index 0 = reference level
    # uses TermSpec.layers as optional hidden stack on top of the embedding


# ---------------------------------------------------------------------------
# Model + training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """A full model = an intercept + a named collection of terms + a link.

    `terms` is keyed by the input name the term consumes (e.g. 'temp',
    'dptp01', 'trend', 'dow'). The model routes inputs to terms by this key,
    so adding or removing terms does not change the forward signature.
    """
    terms: dict[str, TermSpec] = field(default_factory=dict)
    link: Literal["log", "logit", "identity"] = "log"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2500
    lr: float = 8e-4
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    schedule: Literal["cosine", "none"] = "cosine"
    batch_fraction: Optional[float] = None     # None -> full batch
    n_ensemble: int = 3
    loss: Literal["poisson", "bernoulli"] = "poisson"
    grad_clip: Optional[float] = None
    diagnostics_every: Optional[int] = None    # None -> ten diagnostics per fit; 0 disables
    seed: int = 123

    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError("TrainConfig.epochs must be non-negative")
        if self.lr <= 0 or self.lr_min < 0:
            raise ValueError("TrainConfig learning rates must be non-negative, with lr > 0")
        if self.weight_decay < 0:
            raise ValueError("TrainConfig.weight_decay must be non-negative")
        if self.batch_fraction is not None and not 0 < self.batch_fraction <= 1:
            raise ValueError("TrainConfig.batch_fraction must be in (0, 1]")
        if self.schedule not in ("cosine", "none"):
            raise ValueError("TrainConfig.schedule must be 'cosine' or 'none'")
        if self.loss not in ("poisson", "bernoulli"):
            raise ValueError("TrainConfig.loss must be 'poisson' or 'bernoulli'")
        if self.n_ensemble < 1:
            raise ValueError("TrainConfig.n_ensemble must be at least 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("TrainConfig.grad_clip must be positive when set")
        if self.diagnostics_every is not None and self.diagnostics_every < 0:
            raise ValueError("TrainConfig.diagnostics_every must be non-negative")


# ---------------------------------------------------------------------------
# Broadcasting helper — explicit, opt-in convenience
# ---------------------------------------------------------------------------

def broadcast_terms(names: Sequence[str],
                    default: TermSpec,
                    overrides: Optional[dict[str, TermSpec]] = None
                    ) -> dict[str, TermSpec]:
    """Apply one default spec across many names, with per-name overrides.

    Applies one default spec across many names while allowing per-name
    overrides. `replace(...)` is useful for partial overrides.
    """
    overrides = overrides or {}
    return {n: overrides.get(n, default) for n in names}

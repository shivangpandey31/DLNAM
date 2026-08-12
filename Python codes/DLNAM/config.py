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

    lag_max: int = 30

    input_exu: Optional[ExUSpec] = field(
        default_factory=ExUSpec
    )

    def __post_init__(self):
        super().__post_init__()

        if self.lag_max < 0:
            raise ValueError(
                "lag_max must be >= 0"
            )

        if len(self.layers) == 0:
            raise ValueError(
                "SurfaceTermSpec requires at least one LayerSpec"
            )


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
    """Categorical additive term.

    ``encoding_type='one_hot'`` reproduces the original DLNAM categorical
    pathway. ``layers=()`` is a linear lookup on the one-hot design (fixed-effect
    style); non-empty layers add an MLP after the one-hot input.

    ``encoding_type='embedding'`` is the memory-efficient alternative. With
    ``embedding_dim=1`` and ``layers=()`` it is a scalable person/category
    intercept with one learned scalar per level.

    ``source_col`` lets a diagnostic term name (e.g. ``'person'``) read from a
    differently named DataFrame column (e.g. ``'id'``). ``order`` is optional:
    when empty, DataProcessor infers the category order from the supplied data,
    provided the number of levels matches ``num_categories``.
    """
    num_categories: int = 2
    order: Sequence[object] = ()
    source_col: Optional[str] = None
    encoding_type: Literal["one_hot", "embedding"] = "embedding"
    embedding_dim: int = 1
    role: Literal["covariate", "strata"] = "covariate"

    def __post_init__(self):
        super().__post_init__()
        if int(self.num_categories) < 1:
            raise ValueError("CategoricalTermSpec.num_categories must be >= 1")
        if self.encoding_type not in ("one_hot", "embedding"):
            raise ValueError("CategoricalTermSpec.encoding_type must be 'one_hot' or 'embedding'")
        if int(self.embedding_dim) < 1:
            raise ValueError("CategoricalTermSpec.embedding_dim must be >= 1")
        if self.role not in ("covariate", "strata"):
            raise ValueError("CategoricalTermSpec.role must be 'covariate' or 'strata'")
        if self.role == "strata":
            if self.layers:
                raise ValueError("strata terms must use hidden_layers=[] / layers=()")
            if self.encoding_type == "embedding" and int(self.embedding_dim) != 1:
                raise ValueError("embedding strata terms require embedding_dim=1")
        if self.order and len(self.order) != int(self.num_categories):
            raise ValueError(
                "CategoricalTermSpec.order length must equal num_categories when order is supplied"
            )


def _encoding_activation_factory(activation):
    """Normalise the old encoding-config activation field to a fresh-module factory."""
    import copy

    if isinstance(activation, nn.Module):
        return lambda activation=activation: copy.deepcopy(activation)
    if callable(activation):
        return activation
    raise TypeError("encoding config 'activation' must be an nn.Module or callable module factory")


def categorical_terms_from_configs(encoding_configs) -> dict[str, CategoricalTermSpec]:
    """Convert original-style ``encoding_configs`` dictionaries into v2 terms.

    Supported keys:
      required: ``name``, ``num_categories``
      optional: ``col``, ``order``, ``hidden_layers``, ``activation``,
                ``encoding_type``, ``embedding_dim``, ``enabled``, ``role``.

    ``enabled=False`` is a convenience for turning a fixed-effect/categorical
    adjustment off without deleting its configuration.
    """
    out: dict[str, CategoricalTermSpec] = {}
    for i, ec in enumerate(encoding_configs or []):
        if not ec.get("enabled", True):
            continue
        if "name" not in ec or "num_categories" not in ec:
            raise ValueError(
                "each encoding config requires 'name' and 'num_categories'"
            )
        name = str(ec["name"])
        if name in out:
            raise ValueError(f"duplicate encoding config name '{name}'")
        ncat = ec["num_categories"]
        if ncat is None:
            raise ValueError(
                f"encoding config '{name}' has num_categories=None. "
                "V2 must know the number of categories before building the model; "
                "set it from the training data before constructing ModelConfig."
            )
        act = _encoding_activation_factory(ec.get("activation", nn.Mish))
        layers = tuple(
            LayerSpec(width=int(w), activation=ActivationSpec(base=act))
            for w in ec.get("hidden_layers", [])
        )
        out[name] = CategoricalTermSpec(
            num_categories=int(ncat),
            order=tuple(ec.get("order", ()) or ()),
            source_col=ec.get("col", name),
            encoding_type=ec.get("encoding_type", "one_hot"),
            embedding_dim=int(ec.get("embedding_dim", 1)),
            role=ec.get("role", "covariate"),
            layers=layers,
            num_subnets=1,
            scaling="none",
        )
    return out


# ---------------------------------------------------------------------------
# Model + training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """A full model = an intercept + a named collection of terms + a link.

    ``terms`` is the native v2 API. ``encoding_configs`` is a compatibility/
    convenience API retained from v1 so categorical or fixed-effect style terms
    can be toggled without manually constructing CategoricalTermSpec objects.
    Converted categorical terms are merged into ``terms`` during __post_init__.
    """
    terms: dict[str, TermSpec] = field(default_factory=dict)
    link: Literal["log", "logit", "identity"] = "log"
    encoding_configs: Optional[Sequence[dict]] = None
    strata_config: Optional[dict] = None

    def __post_init__(self):
        merged = dict(self.terms)

        # Ordinary estimated categorical covariates (e.g. day of week).
        for name, spec in categorical_terms_from_configs(self.encoding_configs).items():
            if name in merged:
                raise ValueError(
                    f"encoding config '{name}' collides with an existing term name"
                )
            merged[name] = spec

        # Optional explicit nuisance-stratum fixed effect. This is a learned
        # scalar per stratum (sum-to-zero centred in CategoricalTerm), not the
        # conditional elimination performed by gnm(eliminate=...). Keeping it
        # separate from encoding_configs makes the modelling role explicit.
        if self.strata_config and self.strata_config.get("enabled", True):
            sc = dict(self.strata_config)
            sc.setdefault("name", "strata")
            sc.setdefault("col", sc.get("name", "strata"))
            sc.setdefault("encoding_type", "embedding")
            sc.setdefault("embedding_dim", 1)
            sc.setdefault("hidden_layers", [])
            sc["role"] = "strata"
            strata_terms = categorical_terms_from_configs([sc])
            for name, spec in strata_terms.items():
                if name in merged:
                    raise ValueError(
                        f"strata config '{name}' collides with an existing term name"
                    )
                merged[name] = spec

        object.__setattr__(self, "terms", merged)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2500
    lr: float = 8e-4
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    strata_weight_decay: float = 0.0       # nuisance strata are unpenalised by default
    schedule: Literal["cosine", "none"] = "cosine"
    batch_fraction: Optional[float] = None     # None -> full batch
    n_ensemble: int = 3
    loss: Literal["poisson", "bernoulli", "gaussian"] = "poisson"
    grad_clip: Optional[float] = None
    diagnostics_every: Optional[int] = None    # None -> ten diagnostics per fit; 0 disables
    show_progress: bool = True
    gpu_diagnostics: bool = True
    early_stopping: bool = False
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-5
    restore_best_weights: bool = True
    seed: int = 123

    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError("TrainConfig.epochs must be non-negative")
        if self.lr <= 0 or self.lr_min < 0:
            raise ValueError("TrainConfig learning rates must be non-negative, with lr > 0")
        if self.weight_decay < 0:
            raise ValueError("TrainConfig.weight_decay must be non-negative")
        if self.strata_weight_decay < 0:
            raise ValueError("TrainConfig.strata_weight_decay must be non-negative")
        if self.batch_fraction is not None and not 0 < self.batch_fraction <= 1:
            raise ValueError("TrainConfig.batch_fraction must be in (0, 1]")
        if self.schedule not in ("cosine", "none"):
            raise ValueError("TrainConfig.schedule must be 'cosine' or 'none'")
        if self.loss not in ("poisson", "bernoulli", "gaussian"):
            raise ValueError("TrainConfig.loss must be 'poisson', 'bernoulli', or 'gaussian'")
        if self.n_ensemble < 1:
            raise ValueError("TrainConfig.n_ensemble must be at least 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("TrainConfig.grad_clip must be positive when set")
        if self.diagnostics_every is not None and self.diagnostics_every < 0:
            raise ValueError("TrainConfig.diagnostics_every must be non-negative")
        if self.early_stopping_patience < 1:
            raise ValueError("TrainConfig.early_stopping_patience must be at least 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("TrainConfig.early_stopping_min_delta must be non-negative")


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

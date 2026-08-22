# DLNAM — Distributed Lag Non-linear Additive Model

DLNAM is a neural reimagining of the classical **Distributed Lag Non-linear
Model (DLNM)** used in environmental epidemiology (e.g. Gasparrini's
temperature–mortality studies). It replaces the DLNM cross-basis spline with a
**Neural Additive Model (NAM)**-style architecture, keeping the property that
made DLNM useful in the first place: every input's contribution to the
outcome is a separate, plottable, interpretable curve — not a black box.

## Background

Classical DLNM models a lagged exposure–response relationship with a
bidimensional spline "cross-basis" (value × lag) fitted inside a GLM, then
summarised as cumulative or lag-specific relative-risk (RR) curves. DLNAM
targets the same estimand — a value × lag effect surface, plus 1-D
confounder/trend/day-of-week effects — but learns each effect with small
neural subnetworks instead of a fixed spline basis, while preserving DLNM's
centering conventions and reporting scale (log-RR / RR, log-OR / OR) so the
two are directly comparable. Background reading behind this design lives in
`../Documents/` (Gasparrini's DLNM papers, the mgcv/count-data DLNM extension,
and a GAM-vs-DLNM comparison).

## Model

```
eta = intercept + sum_k term_k(x_k)
mu  = link.inverse(eta)          # log -> exp (RR), logit -> sigmoid (OR)
```

Every `term_k` is an `AdditiveTerm` ([dlnam/terms/base.py](dlnam/terms/base.py)) implementing two
things:

- `forward(x)` — its contribution to `eta` during training/prediction.
- `effect(grid, centering)` — a **centered** additive effect curve in raw
  input units, for plotting and for comparison against ground truth or
  against DLNM.

Four term types cover a typical exposure–response study:

| Term | File | Role |
|---|---|---|
| `SurfaceTerm` | [dlnam/terms/surface.py](dlnam/terms/surface.py) | value × lag exposure surface (the DLNM cross-basis analogue) |
| `SmoothTerm` | [dlnam/terms/smooth.py](dlnam/terms/smooth.py) | 1-D smooth confounder (e.g. humidity) |
| `TrendTerm` | [dlnam/terms/smooth.py](dlnam/terms/smooth.py) | long-term time trend, normalised to `[0, 1]` |
| `CategoricalTerm` | [dlnam/terms/categorical.py](dlnam/terms/categorical.py) | categorical effect (e.g. day of week) via a zero-initialised embedding |

The model itself ([dlnam/model.py](dlnam/model.py)) is deliberately thin: intercept + sum of
terms + link. Terms are built from typed, serialisable specs in
[dlnam/config.py](dlnam/config.py), so an experiment's exact architecture is recorded alongside
its results.

### ExU: why the terms stay interpretable

Each subnetwork's first layer is an **Exponential-centred Unit (ExU)**
([dlnam/layers.py](dlnam/layers.py), after Agarwal et al.'s Neural Additive Models):

```
h_j = sum_i exp(w_ij) * (x_i - b_j)
```

The positive, exponentially-parameterised weight plus a learned bias localises
each unit to a region of input space, which is what keeps a NAM subnetwork
from collapsing into an uninterpretable tangle the way an ordinary MLP would.
`SurfaceTerm` needs this in two dimensions (value, lag) at once, so it
supports three encoder strategies ([dlnam/terms/surface.py](dlnam/terms/surface.py)):

- **`concat`** — separate 1-D ExU encoders for value and lag, concatenated.
- **`unified_shared_bias`** — one joint ExU layer, one bias per input
  dimension, shared across units.
- **`unified_local_bias`** — one joint ExU layer, one bias per unit *and* per
  input dimension (reduces exactly to scalar ExU at width 1).

Setting `input_exu=None` gives a plain linear input layer — the "no-ExU"
ablation used to isolate ExU's effect in the benchmark suite.

Within a term, several subnetworks ("subnets") can be mixed by learned
weights (`spec.num_subnets`), an internal ensemble that starts at `1/S` so it
reduces to a single-subnet model at initialisation.

## Uncertainty

Three interval sources, selectable per experiment ([dlnam/inference.py](dlnam/inference.py)):

- **`laplace`** — last-layer Laplace confidence interval from the joint
  Fisher information over the intercept and every term's final linear layer
  ([dlnam/laplace.py](dlnam/laplace.py)); the neural analogue of a DLNM/GLM
  variance–covariance-based CI.
- **`poisson`** — a prediction interval from the outcome's Poisson/quasi-Poisson
  variance.
- **`ensemble`** — spread across ensemble members (a diagnostic, not a
  calibrated coverage statement).

These compose (`laplace+ensemble`) and are all reported on the response scale
(RR/OR) after being built on the link scale.

## Training

[dlnam/train.py](dlnam/train.py) trains a whole ensemble in one vectorised pass with
`torch.func` (`stack_module_state` + `functional_call` + `vmap`), rather than
looping over members. Each member's intercept is warm-started from the target
mean; training supports full-batch or minibatch, cosine LR scheduling,
gradient clipping, and a NaN guard.

## Validation

Two add-on packages keep the core model free of validation-specific code
(one-way dependency: they import `dlnam`, never the reverse):

- **`dlnam_sim`** — synthetic data-generating processes with known ground
  truth, run through a Monte-Carlo study to check bias and interval coverage
  against the true effect curves.
- **`dlnam_bench`** — a head-to-head harness against the R DLNM family
  (QAIC-/QBIC-selected DLNM, penalised P-spline DLNM, tree-based DLNM),
  scored with the *same* centering/grid/scoring logic on both sides so the
  comparison isolates the estimator, not the evaluation.

`experiments/run_real_chicago.py` applies this to the classic Chicago NMMAPS
temperature–mortality dataset (the `dlnm` R package's own example), fitting
DLNAM alongside all four classical comparators and plotting cumulative RR (with
CIs) and the value-by-lag RR contour surface side by side.

## Minimal usage sketch

```python
from dlnam import (
    ModelConfig, SurfaceTermSpec, SmoothTermSpec, TrainConfig,
    DataProcessor, Trainer, EffectExtractor, make_link,
)

cfg = ModelConfig(
    link="log",
    terms={
        "temp": SurfaceTermSpec(lag_max=21, ...),   # value x lag exposure
        "humidity": SmoothTermSpec(...),             # 1-D confounder
    },
)
data = DataProcessor(...).prepare(df)
ensemble = Trainer(TrainConfig(...)).fit(cfg, data)
curve = EffectExtractor(ensemble).effect("temp", interval="laplace+ensemble")
```

See `experiments/run_real_chicago.py` for a complete, working example.

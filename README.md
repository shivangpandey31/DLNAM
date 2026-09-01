# DLNAM — Distributed Lag Neural Additive Models

DLNAM is a neural counterpart to the **Distributed Lag Non-linear Model
(DLNM)** used in environmental epidemiology, as in Gasparrini's
temperature–mortality analyses. It replaces the DLNM's
prespecified spline cross-basis with a **Neural Additive Model (NAM)**-style
architecture while keeping the property that made DLNMs useful in the first
place: every input's contribution to the outcome remains a separate,
plottable, reportable curve rather than a black box.

This repository contains the model, the simulation and benchmarking packages,
and the scripts that produce every result in the accompanying manuscript.
To run it rather than read about it, start at [Install](#install) and
[Reproducing the Paper](#reproducing-the-paper).

## Background

A classical DLNM represents a lagged exposure–response relationship with a
bidimensional spline cross-basis (value × lag) fitted inside a GLM, then
summarised as cumulative or lag-specific relative-risk curves. That
representation has to be specified in advance: basis family, dimension, and
knot placement are chosen or selected for each component, and the cost of
doing so grows with every additional lagged exposure.

DLNAM targets the same estimand, a value × lag effect surface alongside
one-dimensional covariate, trend, and categorical effects, but *learns* each
effect with small neural subnetworks instead of a fixed basis. It preserves
DLNM centering conventions and reporting scales (log-RR / RR, log-OR / OR), so
the two are directly comparable on the same grid.

## Model

```
eta = intercept + sum_k term_k(x_k)
mu  = link.inverse(eta)          # log -> exp (RR), logit -> sigmoid (OR)
```

Every `term_k` is an `AdditiveTerm` ([dlnam/terms/base.py](dlnam/terms/base.py))
implementing two things:

- `forward(x)` — its contribution to `eta` during training and prediction.
- `effect(grid, centering)` — a **centered** additive effect curve in raw
  input units, for plotting and for comparison against ground truth or against
  a DLNM fit.

Four term types cover a typical exposure–response study:

| Term | File | Role |
|---|---|---|
| `SurfaceTerm` | [dlnam/terms/surface.py](dlnam/terms/surface.py) | value × lag exposure surface (the cross-basis analogue) |
| `SmoothTerm` | [dlnam/terms/smooth.py](dlnam/terms/smooth.py) | 1-D smooth covariate (e.g. dew point) |
| `TrendTerm` | [dlnam/terms/smooth.py](dlnam/terms/smooth.py) | long-term time trend, normalised to `[0, 1]` |
| `CategoricalTerm` | [dlnam/terms/categorical.py](dlnam/terms/categorical.py) | categorical effect (e.g. day of week) via a zero-initialised embedding |

The model itself ([dlnam/model.py](dlnam/model.py)) is deliberately thin:
intercept, sum of terms, link. Terms are built from typed, serialisable specs
in [dlnam/config.py](dlnam/config.py), so an experiment's exact architecture is
recorded alongside its results.

### ExU: why the terms stay locally adaptive

Each subnetwork's first layer is an **exp-centered unit (ExU)**
([dlnam/layers.py](dlnam/layers.py), after Agarwal et al.'s Neural Additive
Models):

```
h_j = sum_i exp(w_ij) * (x_i - b_j)
```

The exponentially parameterised weight and learned location bias let each unit
resolve a region of input space, which is what allows a subnetwork to
represent sharp local structure without that flexibility leaking across the
whole domain. `SurfaceTerm` needs this in two dimensions at once (value and
lag), so it supports three encoder strategies
([dlnam/terms/surface.py](dlnam/terms/surface.py)):

- **`concat`** — separate 1-D ExU encoders for value and lag, concatenated.
  This is the reported configuration.
- **`unified_shared_bias`** — one joint ExU layer with one location per input
  coordinate, shared across units.
- **`unified_local_bias`** — one joint ExU layer with one location per unit and
  per coordinate, reducing exactly to the scalar ExU at width 1.

Setting `input_exu=ExUSpec(enabled=False)` gives a plain linear input layer,
the ablation used to isolate ExU's contribution.

Within a term, several subnetworks can be combined by learned mixing weights
(`spec.num_subnets`), an internal mixture initialised from `N(0, 0.1^2)` so the
subnetworks break symmetry at initialisation. Ablating either the ExU layer or
the mixture degrades recovery in every data-generating process tested.

## Uncertainty

Interval sources are selectable per experiment
([dlnam/inference.py](dlnam/inference.py)):

- **`laplace`** — a conditional last-layer Laplace confidence interval built
  from the information over the global intercept and the target component's
  final linear layer ([dlnam/laplace.py](dlnam/laplace.py)); the neural
  analogue of a GLM variance–covariance interval.
- **`ensemble`** — spread across independently initialised ensemble members.
- **`poisson`** — a prediction interval from the outcome's Poisson or
  quasi-Poisson variance.

The first two compose as `laplace+ensemble`, which is what the manuscript
reports: member-level Laplace variance combined with between-member variation
and summarised as a moment-matched Gaussian interval. Intervals are built on
the link scale and reported on the response scale.

## Training

[dlnam/train.py](dlnam/train.py) trains a whole ensemble in one vectorised pass
with `torch.func` (`stack_module_state`, `functional_call`, `vmap`) rather than
looping over members. Each member's intercept is warm-started from the target
mean. Training supports full-batch or minibatch updates, cosine learning-rate
annealing, gradient clipping, and a NaN guard.

## Install

Python 3.10 or newer. From the repository root:

```bash
pip install -e .
```

That is all the model itself needs. Reproducing the paper additionally
requires the `paper` extra and an R installation for the DLNM comparators:

```bash
pip install -e ".[paper]"
```

See [Reproducing the Paper](#reproducing-the-paper) for the R packages and
their versions.

## Minimal Use

```python
import torch

from dlnam import (
    ActivationSpec,
    Centering,
    DataProcessor,
    EffectExtractor,
    EnsembleIntervalUQ,
    ExUSpec,
    LayerSpec,
    ModelConfig,
    SurfaceTermSpec,
    TrainConfig,
    Trainer,
    make_link,
)

model_config = ModelConfig(
    terms={
        "temp": SurfaceTermSpec(
            layers=[LayerSpec(16, ActivationSpec(base=torch.nn.Mish))],
            lag_max=7,
            num_subnets=1,
            input_exu=ExUSpec(enabled=True),
        )
    },
    link="log",
)

train_config = TrainConfig(epochs=100, n_ensemble=3, loss="poisson")
trainer = Trainer(model_config, train_config)
prepared = DataProcessor(model_config).prepare(df, trainer.ensemble, target_col="death")
trainer.fit(prepared.inputs, prepared.y)

extractor = EffectExtractor(
    trainer.ensemble,
    make_link("log"),
    EnsembleIntervalUQ(),
    Centering(method="reference", value=20.0),
)
effect = extractor.extract("temp")
```

For coverage-bearing pointwise confidence intervals, use
`EffectExtractor.with_laplace(...)` after fitting.
`experiments/run_real_chicago.py` is a complete worked example.

## Validation

Two add-on packages keep the core model free of validation-specific code. The
dependency runs one way: they import `dlnam`, never the reverse.

- **`dlnam_sim`** — data-generating processes with known ground truth, run
  through a Monte Carlo study that scores bias, error, and interval coverage
  against the true effect curves.
- **`dlnam_bench`** — a head-to-head harness against the R DLNM family
  (QAIC- and QBIC-selected DLNMs, penalised P-spline DLNM, treed DLNM), scored
  with the *same* centering, grid, and metric code on both sides so the
  comparison isolates the estimator rather than the evaluation.

`experiments/run_real_chicago.py` applies this to the Chicago NMMAPS
temperature–mortality series (the `dlnm` package's own example), fitting DLNAM
alongside all four comparators and plotting the cumulative relative-risk curve
with its intervals beside the value-by-lag response surface.

## Reproducing the Paper

### Environment

```bash
pip install -e ".[paper]"
```

The `paper` extra adds `pyarrow` and `psutil`, which the experiment runners
need but the core model does not. Python 3.10 or newer is required; the
reported runs used Python 3.11 with PyTorch 2.13 and CUDA 12.6 on an NVIDIA
GTX 1070.

The comparators are fitted in R, which must be on `PATH`. The reported runs
used R 4.5.0 for the applications and 4.6.1 for the simulations, with:

| package | version | used by |
| --- | --- | --- |
| `dlnm` | 2.4.10 | all DLNM comparators |
| `mgcv` | 1.9.4 | P-DLNM |
| `dlmtree` | 1.1.1 | T-DLNM |
| `glmmTMB` | 1.1.14 | malaria comparator |
| `jsonlite` | 2.0.0 | R/Python bridge |

```r
install.packages(c("dlnm", "mgcv", "dlmtree", "glmmTMB", "jsonlite"))
```

### Data

The Chicago NMMAPS series ships with the `dlnm` R package and needs no
download. The malaria data are subject to the access conditions of the
original study and are not distributed here, so the malaria runner cannot be
executed without them; every other result reproduces from a clean checkout.

### Order

Each script writes its JSON and figures into `experiments/results/`. The
runners are independent, so they can be run in any order.

```bash
python experiments/run_mc.py            # main comparison,       R = 200
python experiments/run_mc_ablation.py   # architecture ablation, R = 50
python experiments/run_mc_exu.py        # ExU input layers,      R = 50
python experiments/run_mc_joint.py      # joint exposures,       R = 50
python experiments/run_real_chicago.py  # Chicago application
python experiments/run_real_malaria.py  # malaria application (restricted data)
python experiments/run_runtime_scaling.py
python experiments/plot_dgp_surfaces.py
python experiments/make_mixture_figure.py
```

Replicate counts default to the published values. For a fast check of the
pipeline, request a smaller run explicitly rather than editing the defaults:

```bash
python experiments/run_mc.py --n-reps 1 --epochs 100
```

### Cost

Times are for the reported hardware. The simulations are dominated by the R
comparators, the applications by the DLNAM fits.

| run | approximate cost |
| --- | --- |
| `run_mc.py` | ~19 h (12.6 h DLNAM, 6.5 h R) |
| `run_mc_ablation.py`, `run_mc_exu.py`, `run_mc_joint.py` | hours each |
| `run_real_chicago.py` | ~30 min |
| `run_real_malaria.py` | ~41 min |
| `run_runtime_scaling.py` | ~3 h |

Fits are deterministic: the application runners set
`torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG` before
importing PyTorch, so repeated runs on identical hardware and library versions
reproduce the stored results bit for bit.

### Figures Only

`run_mc_joint.py`, `run_real_chicago.py`, `run_real_malaria.py`, and
`run_runtime_scaling.py` redraw their figures from the stored JSON without
refitting:

```bash
python experiments/run_real_chicago.py --figures-only
```

Absolute timings in the scaling benchmark are hardware-dependent, so
re-running it reproduces the procedure rather than the recorded numbers.

## Repository Layout

```text
dlnam/          Installable core package.
dlnam_sim/      Generic DGP utilities and paper simulation scenarios.
dlnam_bench/    R/Python benchmarking bridge for DLNM/T-DLNM comparisons.
experiments/    Paper runners, generated inputs, figures, and result tables.
```

Only `dlnam/` is installed by the package metadata. The simulation and
benchmarking folders are kept in the repository for reproducibility, but
ordinary users of the model do not need them.

## Citation

If you use this software or build on the method, please cite:

```bibtex
@unpublished{helmersson2026dlnam,
  title  = {Distributed Lag Neural Additive Models},
  author = {Helmersson, Calle and Pandey, Shivang and Olivetti, Leonardo
            and Raffetti, Elena},
  year   = {2026}
}
```

To refer to the software rather than the method, cite the tagged release you
used.

## License

MIT.

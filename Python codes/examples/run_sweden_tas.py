"""
run_sweden_tas.py — v2 test runner matching the original Sweden DLNAM v1 setup.

Default v1-compatible design:
  * binary outcome: event
  * logit link + Bernoulli loss
  * temperature distributed-lag surface, lag 0..21
  * pre-lagged input columns: tas_lag0 ... tas_lag21
  * person ``id`` is a learned nuisance STRATUM term (Embedding(n_id, 1))
  * random row minibatching remains valid for this learned-stratum formulation
  * optional ordinary continuous/categorical covariates are configured separately
  * v1-style epoch/GPU diagnostics are printed
  * only plot-relevant CSV files are saved

Important: the neural stratum term estimates one nuisance coefficient per ID. It
is a practical fixed-effect-style analogue, not algebraic gnm(eliminate=id).

Switch INPUT_TYPE to "grouped" if your dataframe contains raw ``tas`` values per
ID/date instead of precomputed lag columns. In grouped mode DataProcessor creates
lags independently inside each ID.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DLNAM import (
    ModelConfig,
    TrainConfig,
    LayerSpec,
    ActivationSpec,
    ExUSpec,
    InitSpec,
    SurfaceTermSpec,
    SmoothTermSpec,
    TrendTermSpec,
    Trainer,
    DataProcessor,
    PerformanceEvaluator,
    ResultVisualizer,
    IntervalUQ,
    make_link,
    Centering,
)


# ---------------------------------------------------------------------------
# SETTINGS — defaults reproduce the original v1 Sweden example as closely as
# is sensible in v2.
# ---------------------------------------------------------------------------

DATA_PATH = Path(r"data_cleaned/sweden_time_series_tas_lagged_weekly.parquet")
OUTPUT_DIR = SCRIPT_DIR / "sweden_tas_results"

TARGET_COL = "event"
ID_COL = "id"
TIME_COL = "date"             # set None if absent in a pre-lagged file
EXPOSURE = "tas"
LAG_MAX = 21

# v1 file is pre-lagged. Change to "grouped" for raw repeated time series.
INPUT_TYPE = "prelagged"       # prelagged | grouped | raw
GROUPBY_COL = ID_COL

SEED = 456
N_ENSEMBLE = 1
N_SUBNETS = 3

FAST_TEST = True
EPOCHS = 20 if FAST_TEST else 1000
BATCH_FRACTION = 0.01
DIAGNOSTICS_EVERY = 10 if FAST_TEST else 20
EARLY_STOPPING_PATIENCE = 5 if FAST_TEST else 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For smoke testing a huge file. None = use the full dataset.
# Sampling is done by complete IDs so repeated rows for sampled people stay together.
SUBSET_N = 100_000 if FAST_TEST else None

USE_TREND = True
SURFACE_STRATEGY = "concat"
SAVE_TRAINING_LOSS_CSV = True

# Ordinary continuous covariates. These are NOT strata.
# Example: ["hurs", "pr"] if those columns exist.
CONTINUOUS_COVARIATES: list[str] = []

# Ordinary categorical covariates. Keep these separate from strata_config.
# Example:
# CATEGORICAL_COVARIATES = [
#     {
#         "name": "dow",
#         "col": "dow",
#         "num_categories": 7,
#         "order": ["Monday", "Tuesday", "Wednesday", "Thursday",
#                   "Friday", "Saturday", "Sunday"],
#         "encoding_type": "one_hot",
#         "hidden_layers": [],
#     }
# ]
CATEGORICAL_COVARIATES: list[dict] = []

LABELS = {
    "tas": "Temperature (°C)",
    "trend": "Time",
    "person": "Person strata",
}


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """DLNAM is file-format agnostic; loaders stay outside the model."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet. "
                "Install pyarrow in your Python environment/container."
            ) from exc
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {suffix}")


def subset_complete_ids(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Approximate a row limit while retaining complete sampled ID strata."""
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    sizes = df.groupby(ID_COL, sort=False).size()
    rng = np.random.default_rng(SEED)
    ids = sizes.index.to_numpy(copy=True)
    rng.shuffle(ids)

    chosen = []
    total = 0
    for ident in ids:
        chosen.append(ident)
        total += int(sizes.loc[ident])
        if total >= max_rows:
            break

    out = df[df[ID_COL].isin(chosen)].copy()
    print(
        f"Subset: retained {len(chosen):,} complete {ID_COL} strata "
        f"({len(out):,} rows; target ~{max_rows:,})"
    )
    return out.reset_index(drop=True)


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {TARGET_COL, ID_COL}

    if INPUT_TYPE == "prelagged":
        required.update(f"{EXPOSURE}_lag{i}" for i in range(LAG_MAX + 1))
    else:
        required.add(EXPOSURE)

    if INPUT_TYPE == "grouped" and TIME_COL is not None:
        required.add(TIME_COL)

    required.update(CONTINUOUS_COVARIATES)
    for ec in CATEGORICAL_COVARIATES:
        if ec.get("enabled", True):
            required.add(ec.get("col", ec["name"]))

    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------

def build_config(df: pd.DataFrame) -> ModelConfig:
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    silu = lambda: ActivationSpec(base=torch.nn.SiLU)
    torch_linear = lambda: InitSpec(scheme="torch_linear")

    # v1 used unconstrained mixing weights initialised near zero.
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)

    terms = {
        EXPOSURE: SurfaceTermSpec(
            lag_max=LAG_MAX,
            layers=[
                LayerSpec(64, mish()),
                LayerSpec(64, mish(), weight_init=torch_linear(), bias_init=torch_linear()),
            ],
            num_subnets=N_SUBNETS,
            scaling="minmax",
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=1.5,
                weight_mean_lag=2.5,
                weight_std=0.5,
                surface_strategy=SURFACE_STRATEGY,
            ),
            constrain_subnet_weights=False,
            mix_init=mix_init(),
            # v1 output_penalty=1e-5 is NOT mathematically identical to v2's
            # term penalty, so do not copy that number mechanically.
            penalty=0.0,
        )
    }

    for name in CONTINUOUS_COVARIATES:
        terms[name] = SmoothTermSpec(
            layers=[LayerSpec(32, silu())],
            num_subnets=N_SUBNETS,
            scaling="zscore",
            mix_init=mix_init(),
        )

    # In v1, trend_layers=None actually created [128,128,64]. Keep that here
    # for architecture replication. Set USE_TREND=False for a genuine no-trend model.
    if USE_TREND:
        terms["trend"] = TrendTermSpec(
            layers=[
                LayerSpec(128, silu()),
                LayerSpec(128, silu()),
                LayerSpec(64, silu()),
            ],
            num_subnets=N_SUBNETS,
            input_exu=ExUSpec(enabled=False),
            constrain_subnet_weights=False,
            mix_init=mix_init(),
        )

    # Neural nuisance stratum: one scalar per ID. Separate from ordinary
    # encoding_configs so its modelling role is explicit and weight decay can
    # be controlled separately.
    strata_config = {
        "name": "person",
        "col": ID_COL,
        "num_categories": int(df[ID_COL].nunique()),
        "order": [],                # infer labels from this training dataframe
        "encoding_type": "embedding",
        "embedding_dim": 1,
        "hidden_layers": [],
        "enabled": True,
    }

    return ModelConfig(
        terms=terms,
        encoding_configs=CATEGORICAL_COVARIATES,
        strata_config=strata_config,
        link="logit",
    )


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SWEDEN TAS — DLNAM V2 / V1-COMPATIBILITY TEST")
    print("=" * 80)
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"Selected device: {DEVICE}")
    print(f"Data path      : {DATA_PATH}")
    print(f"Input type     : {INPUT_TYPE}")
    print(f"Exposure       : {EXPOSURE}; lag 0..{LAG_MAX}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Ensemble       : {N_ENSEMBLE}")
    print(f"Batch fraction : {BATCH_FRACTION}")
    print(f"Trend          : {USE_TREND}")

    df = read_table(DATA_PATH)
    validate_input_columns(df)
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    df = subset_complete_ids(df, SUBSET_N)

    print(f"Rows           : {len(df):,}")
    print(f"ID strata      : {df[ID_COL].nunique():,}")
    print(f"Events         : {df[TARGET_COL].sum():,.0f}")
    print(f"Event rate     : {100 * df[TARGET_COL].mean():.4f}%")

    model_cfg = build_config(df)
    train_cfg = TrainConfig(
        epochs=EPOCHS,
        n_ensemble=N_ENSEMBLE,
        lr=5e-4,
        lr_min=1e-5,
        weight_decay=5e-4,
        strata_weight_decay=0.0,
        schedule="cosine",
        batch_fraction=BATCH_FRACTION,
        loss="bernoulli",
        grad_clip=5.0,
        diagnostics_every=DIAGNOSTICS_EVERY,
        show_progress=True,
        gpu_diagnostics=True,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=1e-5,
        restore_best_weights=True,
        seed=SEED,
    )

    trainer = Trainer(model_cfg, train_cfg, device=torch.device(DEVICE))
    processor = DataProcessor(model_cfg)

    prepare_kwargs = dict(
        target_col=TARGET_COL,
        input_type=INPUT_TYPE,
        fit_scaling=True,
    )
    if INPUT_TYPE == "grouped":
        prepare_kwargs["groupby_col"] = GROUPBY_COL
        prepare_kwargs["time_col"] = TIME_COL
    elif INPUT_TYPE == "prelagged" and TIME_COL is not None and TIME_COL in df.columns:
        prepare_kwargs["time_col"] = TIME_COL

    prepared = processor.prepare(df, trainer.ensemble, **prepare_kwargs)

    print(f"\nPrepared samples: {prepared.n_samples:,}")
    for name, x in prepared.inputs.items():
        role = "strata" if name == "person" else "term"
        print(f"  {name:12s}: {tuple(x.shape)}  [{role}]")

    if prepared.category_orders and "person" in prepared.category_orders:
        print(f"  mapped strata : {len(prepared.category_orders['person']):,}")

    trainer.fit(prepared.inputs, prepared.y)

    print("\nFIT SUMMARY")
    for key, value in trainer.fit_summary.items():
        print(f"  {key:18s}: {value}")

    evaluator = PerformanceEvaluator(trainer.ensemble, distribution="bernoulli")
    metrics = evaluator.evaluate(prepared.inputs, prepared.y)
    evaluator.report(metrics, detailed=True)

    viz = ResultVisualizer(
        trainer.ensemble,
        make_link("logit"),
        IntervalUQ("laplace"),
        Centering(method="median"),
        distribution="bernoulli",
        labels=LABELS,
        trainer=trainer,
        prepared=prepared,
        # Only build Laplace objects for the scientific exposure term. This
        # avoids trying to construct huge uncertainty matrices for ID strata.
        laplace_terms=[EXPOSURE],
    )

    written = viz.save_results_csv(
        OUTPUT_DIR,
        terms=[EXPOSURE],
        alpha=0.05,
        include_surfaces=True,
        include_training_loss=SAVE_TRAINING_LOSS_CSV,
        include_monitor=False,
        include_fit_summary=False,
    )

    print("\nPlot-data CSV outputs:")
    for logical_name, path in written.items():
        print(f"  {logical_name:20s} -> {path}")

    fig, (ax_rr, ax_surface, ax_loss) = plt.subplots(1, 3, figsize=(18, 5))
    viz.plot_effect(EXPOSURE, ax=ax_rr)
    viz.plot_surface(EXPOSURE, ax=ax_surface)
    viz.plot_training_loss(ax=ax_loss)
    fig.suptitle("Sweden temperature — DLNAM v2", fontsize=14)
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "sweden_tas_dlnam_v2.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"\nExecution time: {time.time() - started:.2f} seconds")

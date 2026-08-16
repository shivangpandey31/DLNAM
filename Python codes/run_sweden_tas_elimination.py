"""
Sweden DLNAM runner with three explicit stratum modes.

STRATA_MODE = "eliminate"  (recommended for direct gnm(eliminate=id)-style test)
    * log link + profiled Poisson likelihood
    * ID defines complete matched strata but is NOT a neural input
    * nuisance stratum intercepts are profiled/eliminated analytically
    * mini-batching samples COMPLETE strata, never arbitrary rows
    * output effect is a Relative Risk (RR)

STRATA_MODE = "learned"    (v1-compatible behaviour)
    * logit link + Bernoulli likelihood
    * one learned scalar embedding per ID
    * random row mini-batching
    * output effect is an Odds Ratio (OR)

STRATA_MODE = "none"
    * logit link + Bernoulli likelihood
    * no ID term and no elimination

For the profiled mode, each elimination stratum must contain positive outcome
mass (normally one event among the 4-5 matched candidate days in this study).
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

DATA_PATH = Path(r"Processed/10_sweden_time_series_tas_lagged_weekly.parquet")

# How the matched/person stratum is handled:
#   "eliminate" -> profiled Poisson, conceptually aligned with gnm(eliminate=id)
#   "learned"   -> current/v1 learned scalar ID embedding
#   "none"      -> no stratum adjustment
STRATA_MODE = "eliminate"

OUTPUT_DIR = SCRIPT_DIR / f"sweden_tas_results_{STRATA_MODE}"

TARGET_COL = "event"
ID_COL = "LopNr_PersonNr"
# For the current one-event-per-person dataset this is the same column. If a
# person can contribute multiple matched event sets, replace this with a unique
# matched-set column (for example person x event episode/month).
ELIMINATE_COL = ID_COL
TIME_COL = "date"             # set None if absent in a pre-lagged file
EXPOSURE = "tas"
LAG_MAX = 21

# v1 file is pre-lagged. Change to "grouped" for raw repeated time series.
INPUT_TYPE = "prelagged"       # prelagged | grouped | raw
GROUPBY_COL = ID_COL

SEED = 123
N_ENSEMBLE = 1
N_SUBNETS = 3

FAST_TEST = True
EPOCHS = 1000 if FAST_TEST else 1000
BATCH_FRACTION = 0.1
DIAGNOSTICS_EVERY = 1 if FAST_TEST else 10
EARLY_STOPPING_PATIENCE = 100 if FAST_TEST else 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For smoke testing a huge file. None = use the full dataset.
# Sampling is done by complete IDs so repeated rows for sampled people stay together.
SUBSET_N = 50_000 if FAST_TEST else None

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

def read_table(path: Path,) -> pd.DataFrame:
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


def subset_complete_ids(df: pd.DataFrame, subset_n: int | None, groupby_col: str | None = None) -> pd.DataFrame:
    """
    Subset dataframe while preserving complete groups when groupby_col is given.
    If groupby_col is provided: subset_n = number of unique groups to retain.
    If groupby_col is None: subset_n = number of rows to retain.
    """

    if subset_n is None:
        return df.reset_index(drop=True)

    # ---------------------------------------------------------
    # Grouped data: sample complete groups
    # ---------------------------------------------------------
    if groupby_col is not None:

        if groupby_col not in df.columns:
            raise ValueError(f"groupby_col='{groupby_col}' not found in dataframe.")

        groups = df[groupby_col].dropna().unique()

        if len(groups) <= subset_n:
            print(f"Subset not required: dataframe contains {len(groups):,} unique {groupby_col} groups.")
            return df.reset_index(drop=True)

        rng = np.random.default_rng(SEED)
        selected_groups = rng.choice(groups, size=subset_n, replace=False)
        out = df[df[groupby_col].isin(selected_groups)].copy()
        print(f"Subset: retained {len(selected_groups):,} complete {groupby_col} groups ({len(out):,} rows).")

        return out.reset_index(drop=True)

    # ---------------------------------------------------------
    # Ungrouped data: sample rows
    # ---------------------------------------------------------
    if len(df) <= subset_n:
        return df.reset_index(drop=True)

    out = df.sample(n=subset_n, random_state=SEED).copy()
    print(f"Subset: retained {len(out):,} rows.")

    return out.reset_index(drop=True)

def validate_input_columns(df: pd.DataFrame) -> None:
    required = {TARGET_COL, ID_COL}
    if STRATA_MODE == "eliminate":
        required.add(ELIMINATE_COL)

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

    if STRATA_MODE not in {"eliminate", "learned", "none"}:
        raise ValueError(
            "STRATA_MODE must be 'eliminate', 'learned', or 'none'"
        )

    # Only the v1-compatible learned mode creates a neural ID term. In
    # elimination mode the ID is passed to the likelihood separately and never
    # enters the neural network.
    strata_config = None
    if STRATA_MODE == "learned":
        strata_config = {
            "name": "person",
            "col": ID_COL,
            "num_categories": int(df[ID_COL].nunique()),
            "order": [],
            "encoding_type": "embedding",
            "embedding_dim": 1,
            "hidden_layers": [],
            "enabled": True,
        }

    link = "log" if STRATA_MODE == "eliminate" else "logit"

    return ModelConfig(
        terms=terms,
        encoding_configs=CATEGORICAL_COVARIATES,
        strata_config=strata_config,
        link=link,
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
    print(f"Strata mode    : {STRATA_MODE}")

    df = read_table(DATA_PATH)
    validate_input_columns(df)
    df[TARGET_COL] = df[TARGET_COL].astype(float)

    subset_group_col = ELIMINATE_COL if STRATA_MODE == "eliminate" else GROUPBY_COL
    df = subset_complete_ids(df, SUBSET_N, subset_group_col)

    print(f"Rows           : {len(df):,}")
    print(f"ID strata      : {df[ID_COL].nunique():,}")
    print(f"Events         : {df[TARGET_COL].sum():,.0f}")
    print(f"Event rate     : {100 * df[TARGET_COL].mean():.4f}%")

    if STRATA_MODE == "eliminate":
        group_rows = df.groupby(ELIMINATE_COL, sort=False).size()
        group_events = df.groupby(ELIMINATE_COL, sort=False)[TARGET_COL].sum()
        print(
            f"Rows/stratum   : min {group_rows.min():,}; "
            f"median {group_rows.median():.1f}; max {group_rows.max():,}"
        )
        print(
            f"Events/stratum : min {group_events.min():.0f}; "
            f"median {group_events.median():.1f}; max {group_events.max():.0f}"
        )
        if (group_events <= 0).any():
            raise ValueError(
                "Elimination mode requires positive event mass in every stratum. "
                "Check that ID_COL identifies the complete matched event set."
            )
        if (group_events != 1).any():
            print(
                "WARNING: not every elimination stratum has exactly one event. "
                "The profiled Poisson likelihood supports positive stratum totals, "
                "but for your intended one-case matched design verify that "
                "ELIMINATE_COL identifies one event episode/matched set rather "
                "than only the person."
            )

    model_cfg = build_config(df)
    train_loss = "profiled_poisson" if STRATA_MODE == "eliminate" else "bernoulli"
    train_cfg = TrainConfig(
        epochs=EPOCHS,
        n_ensemble=N_ENSEMBLE,
        lr=5e-4,
        lr_min=1e-5,
        weight_decay=5e-4,
        strata_weight_decay=0.0,
        schedule="cosine",
        batch_fraction=BATCH_FRACTION,
        loss=train_loss,
        grad_clip=5.0,
        diagnostics_every=DIAGNOSTICS_EVERY,
        show_progress=True,
        gpu_diagnostics=True,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=1e-4,
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
    if STRATA_MODE == "eliminate":
        prepare_kwargs["eliminate_col"] = ELIMINATE_COL
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

    if prepared.eliminate_index is not None:
        print(
            f"  eliminated strata: "
            f"{int(prepared.eliminate_index.max().item()) + 1:,} "
            f"(column: {prepared.eliminate_col})"
        )

    if prepared.category_orders and "person" in prepared.category_orders:
        print(f"  mapped strata : {len(prepared.category_orders['person']):,}")

    trainer.fit(
        prepared.inputs,
        prepared.y,
        eliminate_index=prepared.eliminate_index,
    )

    print("\nFIT SUMMARY")
    for key, value in trainer.fit_summary.items():
        print(f"  {key:18s}: {value}")

    if STRATA_MODE != "eliminate":
        evaluator = PerformanceEvaluator(trainer.ensemble, distribution="bernoulli")
        metrics = evaluator.evaluate(prepared.inputs, prepared.y)
        evaluator.report(metrics, detailed=True)
    else:
        print(
            "\nPredictive outcome-scale diagnostics skipped: profiled elimination "
            "does not estimate the stratum intercepts required for unconditional "
            "mean predictions. The exposure RR remains identifiable."
        )

    link_name = "log" if STRATA_MODE == "eliminate" else "logit"
    distribution = "poisson" if STRATA_MODE == "eliminate" else "bernoulli"

    # In elimination mode there is no huge learned person-ID term, so the
    # Laplace covariance can include all fitted terms (e.g. trend/covariates)
    # jointly. In learned mode keep the exposure-only Laplace object to avoid a
    # massive ID-stratum covariance block.
    laplace_terms = None if STRATA_MODE == "eliminate" else [EXPOSURE]

    viz = ResultVisualizer(
        trainer.ensemble,
        make_link(link_name),
        IntervalUQ("laplace"),
        Centering(method="median"),
        distribution=distribution,
        labels=LABELS,
        trainer=trainer,
        prepared=prepared,
        laplace_terms=laplace_terms,
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
    fig.suptitle(f"Sweden temperature — DLNAM ({STRATA_MODE})", fontsize=14)
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "sweden_tas_dlnam.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"\nExecution time: {time.time() - started:.2f} seconds")

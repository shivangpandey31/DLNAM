"""
run_chicago.py — Chicago NMMAPS integration test for DLNAM v2.

This example exercises the current v2 architecture while keeping modelling roles
explicit:
  * ``temp`` is a distributed-lag SurfaceTerm.
  * dew point / ozone / PM10 are continuous SmoothTerms.
  * ``dow`` is an ordinary categorical covariate via ``encoding_configs``.
  * Chicago is one continuous daily series, therefore ``input_type='raw'``.
  * v1-style epoch + GPU-memory diagnostics are printed by Trainer.
  * only plot-relevant CSV files are written by default.

The DLNAM package receives a pandas DataFrame; file reading stays upstream.
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
    SoftCapSpec,
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
# SETTINGS
# ---------------------------------------------------------------------------

DATA_PATH = Path(r"Python codes/chicago_nmmaps.csv")
OUTPUT_DIR = SCRIPT_DIR / "chicago_results"

LAG_MAX = 30
SEED = 123
N_ENSEMBLE = 1

FAST_TEST = False  # True for quick test runs, False for full training.
EPOCHS = 100 if FAST_TEST else 3500
EARLY_STOPPING_PATIENCE = 20 if FAST_TEST else 100

BATCH_FRACTION = None        # Chicago is small enough for full-batch training.
DIAGNOSTICS_EVERY = 1       # v1-style epoch/GPU print interval.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SURFACE_STRATEGY = "concat"  # concat | unified_shared_bias | unified_local_bias
USE_TREND = True
USE_DOW_COVARIATE = True
DOW_ENCODING_TYPE = "one_hot"  # one_hot | embedding
SAVE_TRAINING_LOSS_CSV = True

DOW_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

LABELS = {
    "temp": "Temperature",
    "dptp01": "Dew Point",
    "o301": "Ozone",
    "pm1001": "PM10",
    "trend": "Time",
    "dow": "Day of Week",
}


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet upstream of DLNAM."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet. "
                "Install pyarrow in the environment/container."
            ) from exc
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {suffix}")


def load_chicago(path: Path) -> pd.DataFrame:
    df = read_table(path)

    # Dataset-specific feature engineering stays outside DataProcessor.
    for col in ["dptp", "o3", "pm10"]:
        df[f"{col}01"] = df[col].rolling(window=2).mean()

    return df


# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------

def build_config() -> ModelConfig:

    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    silu = lambda: ActivationSpec(base=torch.nn.SiLU)

    S = 3

    # v1-like mixing-weight initialisation. Native v2 default is constant 1/S.
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    torch_linear = lambda: InitSpec(scheme="torch_linear")

    terms = {
        "temp": SurfaceTermSpec(
            layers=[
                LayerSpec(128, mish()),
                LayerSpec(
                    128,
                    mish(),
                    weight_init=torch_linear(),
                    bias_init=torch_linear(),
                ),
            ],
            num_subnets=S,
            scaling="minmax",
            lag_max=LAG_MAX,
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=1.5,
                weight_mean_lag=2.5,
                weight_std=0.5,
                surface_strategy=SURFACE_STRATEGY,
                bias_init=exu_bias(),
            ),
            mix_init=mix_init(),
            constrain_subnet_weights=False,
        ),
        "dptp01": SmoothTermSpec(
            layers=[LayerSpec(32, silu(), weight_init=torch_linear(), bias_init=torch_linear())],
            num_subnets=S,
            scaling="zscore",
            mix_init=mix_init(),
        ),
        "o301": SmoothTermSpec(
            layers=[LayerSpec(32, silu(), weight_init=torch_linear(), bias_init=torch_linear())],
            num_subnets=S,
            scaling="zscore",
            mix_init=mix_init(),
        ),
        "pm1001": SmoothTermSpec(
            layers=[LayerSpec(32, silu(), weight_init=torch_linear(), bias_init=torch_linear())],
            num_subnets=S,
            scaling="zscore",
            mix_init=mix_init(),
        ),
    }

    if USE_TREND:
        terms["trend"] = TrendTermSpec(
            layers=[
                LayerSpec(128, silu()),
                LayerSpec(128, silu(), weight_init=torch_linear(), bias_init=torch_linear()),
                LayerSpec(128, silu(), weight_init=torch_linear(), bias_init=torch_linear()),
            ],
            num_subnets=S,
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=4.5,
                weight_std=0.5,
                bias_init=exu_bias(),
            ),
            mix_init=mix_init(),
            constrain_subnet_weights=False,
        )

    encoding_configs = [
        {
            "name": "dow",
            "col": "dow",
            "num_categories": 7,
            "order": DOW_ORDER,
            "hidden_layers": [],
            "activation": torch.nn.Mish,
            "encoding_type": DOW_ENCODING_TYPE,
            "embedding_dim": 1,
            "enabled": USE_DOW_COVARIATE,
        }
    ]

    return ModelConfig(
        terms=terms,
        encoding_configs=encoding_configs,
        link="log",
    )


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHICAGO NMMAPS — DLNAM V2 TEST")
    print("=" * 80)
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"Selected device: {DEVICE}")
    print(f"Data path      : {DATA_PATH}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Ensemble       : {N_ENSEMBLE}")
    print(f"Surface        : {SURFACE_STRATEGY}")
    print(f"DOW covariate  : {USE_DOW_COVARIATE}")

    df = load_chicago(DATA_PATH)
    model_cfg = build_config()

    train_cfg = TrainConfig(
        epochs=EPOCHS,
        n_ensemble=N_ENSEMBLE,
        lr=5e-4,
        lr_min=1e-4,
        weight_decay=1e-4,
        schedule="cosine",
        batch_fraction=BATCH_FRACTION,
        loss="poisson",
        grad_clip=None,
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

    # Chicago is one continuous daily series, not grouped longitudinal data.
    prepared = processor.prepare(
        df,
        trainer.ensemble,
        target_col="death",
        input_type="raw",
        fit_scaling=True,
    )

    print(f"\nPrepared samples: {prepared.n_samples:,}")
    for name, x in prepared.inputs.items():
        print(f"  {name:10s}: {tuple(x.shape)}")

    trainer.fit(prepared.inputs, prepared.y)

    print("\nFIT SUMMARY")
    for key, value in trainer.fit_summary.items():
        print(f"  {key:18s}: {value}")

    evaluator = PerformanceEvaluator(trainer.ensemble, distribution="poisson")
    metrics = evaluator.evaluate(prepared.inputs, prepared.y)
    evaluator.report(metrics, detailed=True)

    viz = ResultVisualizer(
        trainer.ensemble,
        make_link("log"),
        IntervalUQ("laplace"),
        Centering(method="median"),
        distribution="poisson",
        labels=LABELS,
        trainer=trainer,
        prepared=prepared,
    )

    # Save only numerical files needed to recreate the requested plots later.
    written = viz.save_results_csv(
        OUTPUT_DIR,
        terms=["temp"],
        alpha=0.05,
        include_surfaces=True,
        include_training_loss=SAVE_TRAINING_LOSS_CSV,
        include_monitor=False,
        include_fit_summary=False,
    )

    print("\nPlot-data CSV outputs:")
    for logical_name, path in written.items():
        print(f"  {logical_name:20s} -> {path}")

    fig, (ax_temp, ax_surface, ax_loss) = plt.subplots(1, 3, figsize=(18, 5))
    viz.plot_effect("temp", ax=ax_temp)
    viz.plot_surface("temp", ax=ax_surface)
    viz.plot_training_loss(ax=ax_loss)
    fig.suptitle("Chicago NMMAPS — DLNAM v2", fontsize=14)
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "chicago_dlnam_v2.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"\nExecution time: {time.time() - started:.2f} seconds")

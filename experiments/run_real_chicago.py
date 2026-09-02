"""
run_real_chicago.py -- one-shot Chicago DLNAM vs DLNM comparison.

Fits the Chicago NMMAPS temperature-mortality association once with:
  * DLNAM, using the same surface architecture/training settings as the MC code;
  * classical DLNM selected by QAIC;
  * classical DLNM selected by QBIC;
  * penalised P-spline DLNM;
  * tree-based DLNM.

The output is a 2x5 figure: cumulative RR with confidence intervals on the top
row and fitted value-by-lag RR contours on the bottom row. DLNAM uses the
last-layer Laplace CI; the R comparators use their fitted-model summaries.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Deterministic GPU reductions. Several backward kernels (notably the
# categorical embedding) accumulate with atomics, so an identical seed does
# not otherwise reproduce a fit. Must be set before torch initialises cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from dlnam import (
    ActivationSpec,
    Centering,
    CategoricalTermSpec,
    DataProcessor,
    EffectExtractor,
    ExUSpec,
    InitSpec,
    LayerSpec,
    ModelConfig,
    PerformanceEvaluator,
    SmoothTermSpec,
    SurfaceTermSpec,
    TrainConfig,
    Trainer,
    TrendTermSpec,
    make_link,
)

from dlnam_bench import plots as bp
from experiment_io import load_json_if_exists, results_dir, save_json


# ----------------------------- SETTINGS -----------------------------------
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CSV_PATH = ROOT / "chicago_nmmaps.csv"
BENCH_DIR = HERE / "real_chicago_data"
RESULTS_DIR = results_dir(HERE)
RESULT_JSON = RESULTS_DIR / "chicago_model_comparison.json"
RSCRIPT = "Rscript"
RSCRIPT_FILE = ROOT / "dlnam_bench" / "fit_chicago.R"

LAG_MAX = 30
N_GRID = 200
CI_LEVEL = 0.95
REFERENCE = "median"  # use "median" or a fixed value such as 21.0
EPOCHS = 5000
# Curvature penalty applied to the exposure-lag surface.
ROUGHNESS = 0.003
N_ENSEMBLE = 3
SEED = 0
# One interval construction across every analysis: the last-layer
# Laplace variance plus the between-member ensemble spread.
SE_SOURCE = "laplace+ensemble"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Cross-basis search grid for the criterion-selected DLNMs. The penalised
# dimensions take the upper end of the grid, capped by lag_max for the lag
# basis.
VALUE_DF_GRID = tuple(range(2, 6))
LAG_DF_GRID = tuple(range(2, 6))
PENALIZED_VALUE_DF = max(VALUE_DF_GRID)
PENALIZED_LAG_DF = max(LAG_DF_GRID)
# T-DLNM sampler settings, as in the simulation.
TDLNM_SETTINGS = {
    "burn": 5000,
    "iter": 15000,
    "thin": 10,
    "attempts": 10,
    "exposure_splits": 30,
    "trees": 20,
}
SURFACE_SCALE_MODE = "per_model"  # "shared" or "per_model"
# Controls the RR (vertical) axis of the surface row only; the temperature and
# lag axes are identical across panels either way.
# "shared" pools the RR axis over the five fits, making surface amplitude
# comparable across panels -- these span roughly sixfold -- at the cost of
# flattening the smaller surfaces. "per_model" keeps each panel readable.

# Adjustment structure used by the existing Chicago DLNM runner.
CONFOUNDER_SPEC = {"dptp01": 3, "o301": 0, "pm1001": 0}
TREND_DF_PER_YR = 7
DOW_COL = "dow"

CUM_NAME = "chicago_temp_cum.csv"
SURF_NAME = "chicago_temp_surf.csv"
METHODS = {
    "DLNAM": ("", bp.LABELS["DLNAM"]),
    "QAIC": ("", bp.LABELS["QAIC"]),
    "QBIC": ("qbic_", bp.LABELS["QBIC"]),
    "Penalised": ("pen_", bp.LABELS["Penalised"]),
    "TDLNM": ("tree_", bp.LABELS["TDLNM"]),
}
# --------------------------------------------------------------------------


def _hardware() -> dict:
    """Machine the fit actually ran on, so the reported environment is recorded
    rather than asserted. Mirrors the block written by the scaling benchmark."""
    import platform
    hw = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw

def load_chicago(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["dptp", "o3", "pm10"]:
        df[f"{col}01"] = df[col].rolling(window=2).mean()
    return df


def build_dlnam_config() -> ModelConfig:
    """Chicago model with the MC surface architecture and existing adjustments."""
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl = lambda: InitSpec(scheme="torch_linear")

    return ModelConfig(terms={
        "temp": SurfaceTermSpec(
            layers=[
                LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl()),
            ],
            num_subnets=N_ENSEMBLE,
            scaling="minmax",
            lag_max=LAG_MAX,
            roughness_value=ROUGHNESS,
            roughness_lag=ROUGHNESS,
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=1.5,
                weight_mean_lag=2.5,
                weight_std=0.5,
                surface_strategy="concat",
                bias_init=exu_bias(),
            ),
            mix_init=mix_init(),
        ),
        "dptp01": SmoothTermSpec(
            layers=[LayerSpec(32, mish(), weight_init=tl(), bias_init=tl())],
            num_subnets=N_ENSEMBLE,
            scaling="zscore",
            mix_init=mix_init(),
        ),
        "o301": SmoothTermSpec(
            layers=[LayerSpec(32, mish(), weight_init=tl(), bias_init=tl())],
            num_subnets=N_ENSEMBLE,
            scaling="zscore",
            mix_init=mix_init(),
        ),
        "pm1001": SmoothTermSpec(
            layers=[LayerSpec(32, mish(), weight_init=tl(), bias_init=tl())],
            num_subnets=N_ENSEMBLE,
            scaling="zscore",
            mix_init=mix_init(),
        ),
        "trend": TrendTermSpec(
            layers=[
                LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl()),
            ],
            num_subnets=N_ENSEMBLE,
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=4.5,
                weight_std=0.5,
                bias_init=exu_bias(),
            ),
            mix_init=mix_init(),
        ),
        "dow": CategoricalTermSpec(num_categories=7, order=DOW_ORDER),
    }, link="log")


def _calendar_time(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        return df["time"].astype(float)
    if "date" in df.columns:
        date = pd.to_datetime(df["date"])
        return (date - date.min()).dt.days.astype(float)
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _n_years(df: pd.DataFrame) -> int:
    if "year" in df.columns:
        return int(df["year"].max() - df["year"].min() + 1)
    if "date" in df.columns:
        date = pd.to_datetime(df["date"])
        return max(1, int(round((date.max() - date.min()).days / 365.25)))
    return max(1, int(round(len(df) / 365.25)))


def _resolve_reference(temp: np.ndarray) -> float:
    if isinstance(REFERENCE, str):
        if REFERENCE.lower() == "median":
            return float(np.median(temp))
        raise ValueError("REFERENCE must be 'median' or a numeric value")
    return float(REFERENCE)


def prepare_chicago_bench() -> tuple[pd.DataFrame, np.ndarray, float, int]:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    df = load_chicago(CSV_PATH)
    df["caltime"] = _calendar_time(df)
    n_years = _n_years(df)

    keep = ["temp", "death", DOW_COL, "caltime"] + list(CONFOUNDER_SPEC.keys())
    sub = df[keep].dropna().reset_index(drop=True)
    sub.to_csv(BENCH_DIR / "chicago_data.csv", index=False)

    temp = sub["temp"].to_numpy(dtype=float)
    grid = np.linspace(float(temp.min()), float(temp.max()), N_GRID)
    reference = _resolve_reference(temp)
    config = {
        "data": "chicago_data.csv",
        "exposure_col": "temp",
        "target_col": "death",
        "grid": grid.tolist(),
        "reference": reference,
        "lag_max": int(LAG_MAX),
        "ci_level": float(CI_LEVEL),
        "value_df_grid": list(VALUE_DF_GRID),
        "lag_df_grid": list(LAG_DF_GRID),
        "penalized_value_df": int(PENALIZED_VALUE_DF),
        "penalized_lag_df": int(PENALIZED_LAG_DF),
        "tdlnm_burn": int(TDLNM_SETTINGS["burn"]),
        "tdlnm_iter": int(TDLNM_SETTINGS["iter"]),
        "tdlnm_thin": int(TDLNM_SETTINGS["thin"]),
        "tdlnm_attempts": int(TDLNM_SETTINGS["attempts"]),
        "tdlnm_exposure_splits": int(TDLNM_SETTINGS["exposure_splits"]),
        "tdlnm_trees": int(TDLNM_SETTINGS["trees"]),
        "tdlnm_seed": int(SEED),
        "confounder_spec": CONFOUNDER_SPEC,
        "trend_df": int(TREND_DF_PER_YR * n_years),
        "dow_col": DOW_COL,
        "time_col": "caltime",
        "cum_name": CUM_NAME,
        "surf_name": SURF_NAME,
    }
    with open(BENCH_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Data")
    print(f"  complete cases {len(sub)}")
    print(f"  lag maximum    {LAG_MAX}")
    print(f"  reference temp {reference:.2f}")
    print(f"  trend df       {config['trend_df']} ({n_years} years)")
    return sub, grid, reference, n_years


def run_dlnm_fits() -> None:
    cmd = [RSCRIPT, str(RSCRIPT_FILE), str(BENCH_DIR), "qaic,qbic,pen,tdlnm"]
    print("\nDLNM fits")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print("  command failed: " + " ".join(cmd))
        print(result.stderr)
        raise SystemExit(f"R exited with status {result.returncode}")


def fit_dlnam(df: pd.DataFrame, grid: np.ndarray, reference: float) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = build_dlnam_config()
    train_config = TrainConfig(
        epochs=EPOCHS,
        n_ensemble=N_ENSEMBLE,
        lr=8e-4,
        lr_min=1e-4,
        weight_decay=1e-4,
        schedule="cosine",
        grad_clip=10,
        seed=SEED,
    )
    trainer = Trainer(config, train_config, device=torch.device(DEVICE))
    prepared = DataProcessor(config).prepare(df, trainer.ensemble, target_col="death")
    print("\nDLNAM fit")
    print(f"  samples  {prepared.n_samples}")
    print(f"  ensemble {N_ENSEMBLE}")
    print(f"  device   {DEVICE}")
    trainer.fit(prepared.inputs, prepared.y)

    evaluator = PerformanceEvaluator(trainer.ensemble, distribution="poisson")
    perf = evaluator.evaluate(prepared.inputs, prepared.y)
    print()
    evaluator.report(perf)

    link = make_link("log")
    centering = Centering(method="reference", value=reference)
    extractor = EffectExtractor.with_laplace(
        trainer.ensemble, prepared, link, centering, interval=SE_SOURCE
    )
    estimate = extractor.extract("temp", grid, alpha=1.0 - CI_LEVEL)
    ref = float(reference)
    surface_rr = np.exp(np.mean([
        model.term("temp").per_lag_log_rr(grid, ref)
        for model in trainer.ensemble
    ], axis=0))
    lags = np.arange(surface_rr.shape[0])
    return {
        "curve": pd.DataFrame({
            "value": grid,
            "fit": estimate.mean,
            "lo": estimate.lo,
            "hi": estimate.hi,
            "log_mean": estimate.log_mean,
            "log_se": extractor.last_laplace_components["se_total"],
        }),
        "surface": pd.DataFrame({
            "value": np.tile(grid, len(lags)),
            "lag": np.repeat(lags, len(grid)),
            "rr": surface_rr.reshape(-1),
        }),
        "performance": perf,
        "laplace": {
            "phi": float(extractor.phi),
            "prior_precision": float(extractor.last_prior_precision),
        },
        "fit_summary": trainer.fit_summary,
    }


def load_dlnm_curve(prefix: str) -> pd.DataFrame:
    path = BENCH_DIR / f"{prefix}{CUM_NAME}"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_dlnm_surface(prefix: str) -> pd.DataFrame:
    path = BENCH_DIR / f"{prefix}{SURF_NAME}"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_outputs(dlnam: dict, dlnm_curves: dict[str, pd.DataFrame],
                 dlnm_surfaces: dict[str, pd.DataFrame],
                 grid: np.ndarray, reference: float, n_years: int) -> dict:
    dlnam["curve"].to_csv(BENCH_DIR / "dlnam_chicago_temp_cum.csv", index=False)
    dlnam["surface"].to_csv(BENCH_DIR / "dlnam_chicago_temp_surf.csv", index=False)
    curves = {
        "DLNAM": dlnam["curve"],
        **dlnm_curves,
    }
    surfaces = {
        "DLNAM": dlnam["surface"],
        **dlnm_surfaces,
    }
    summary = {
        "kind": "chicago_model_comparison",
        "models": list(METHODS.keys()),
        "settings": {
            "lag_max": LAG_MAX,
            "n_grid": N_GRID,
            "ci_level": CI_LEVEL,
            "se_source": SE_SOURCE,
            "epochs": EPOCHS,
            "n_ensemble": N_ENSEMBLE,
            "seed": SEED,
            "device": DEVICE,
            "reference": reference,
            "value_df_grid": list(VALUE_DF_GRID),
            "lag_df_grid": list(LAG_DF_GRID),
            "penalized_value_df": int(PENALIZED_VALUE_DF),
            "penalized_lag_df": int(PENALIZED_LAG_DF),
            "tdlnm_family": "gaussian_log1p",
            "tdlnm_burn": int(TDLNM_SETTINGS["burn"]),
            "tdlnm_iter": int(TDLNM_SETTINGS["iter"]),
            "tdlnm_thin": int(TDLNM_SETTINGS["thin"]),
            "tdlnm_attempts": int(TDLNM_SETTINGS["attempts"]),
            "tdlnm_exposure_splits": int(TDLNM_SETTINGS["exposure_splits"]),
            "tdlnm_trees": int(TDLNM_SETTINGS["trees"]),
            "tdlnm_seed": int(SEED),
            "trend_df": int(TREND_DF_PER_YR * n_years),
            "dlnam_ci": "last_layer_laplace",
            "dlnm_ci": "crosspred_or_tdlnm_summary",
        },
        "dlnam_performance": dlnam["performance"],
        "dlnam_laplace": dlnam["laplace"],
        "dlnam_fit_summary": dlnam["fit_summary"],
        "r_environment": load_json_if_exists(BENCH_DIR / "r_environment.json"),
        "hardware": _hardware(),
        "tdlnm_fit_status": load_json_if_exists(BENCH_DIR / "tdlnm_fit_status.json"),
        "pen_diagnostics": load_json_if_exists(BENCH_DIR / "pen_diagnostics.json"),
        "curves": {
            name: {
                "value": df["value"].tolist(),
                "fit": df["fit"].tolist(),
                "lo": df["lo"].tolist(),
                "hi": df["hi"].tolist(),
            }
            for name, df in curves.items()
        },
        "surfaces": {
            name: {
                "value": df["value"].tolist(),
                "lag": df["lag"].tolist(),
                "rr": df["rr"].tolist(),
            }
            for name, df in surfaces.items()
        },
    }
    save_json(RESULT_JSON, summary)
    return summary


ORDER = ["DLNAM", "QAIC", "QBIC", "Penalised", "TDLNM"]
OUT_STEM = "chicago_model_comparison"


PLOT_RC = {
    **bp._RC,
    "axes.grid": False,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
}


def _as_frame(entry: dict) -> pd.DataFrame:
    return pd.DataFrame({key: np.asarray(value) for key, value in entry.items()})


def _surface_matrix(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _as_frame(entry)
    pivot = (
        df.groupby(["lag", "value"], as_index=False)["rr"].mean()
        .pivot(index="lag", columns="value", values="rr")
        .sort_index()
    )
    values = pivot.columns.to_numpy(dtype=float)
    lags = pivot.index.to_numpy(dtype=float)
    rr = pivot.to_numpy(dtype=float)
    return values, lags, rr


def _lightened(colour: str, amount: float = 0.40) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(colour), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def _surface_facecolors(rr: np.ndarray, colour: str) -> np.ndarray:
    """Single-hue model surface, with height shown by lightness."""
    base = np.asarray(to_rgb(colour), dtype=float)
    light = np.asarray(_lightened(colour, 0.74), dtype=float)
    z = np.asarray(rr, dtype=float)
    lo, hi = np.nanpercentile(z, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros_like(z)
    else:
        scaled = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    # Gentle nonlinear contrast: low RR stays pale, high RR reaches the model colour.
    scaled = scaled ** 0.75
    rgb = light[None, None, :] * (1.0 - scaled[..., None]) + base[None, None, :] * scaled[..., None]
    alpha = np.full((*z.shape, 1), 0.92)
    return np.concatenate([rgb, alpha], axis=-1)


def _upsample_surface(
    values: np.ndarray,
    lags: np.ndarray,
    rr: np.ndarray,
    *,
    n_values: int = 640,
    n_lags: int = 440,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly upsample a fitted surface for smoother 3D rendering only."""
    dense_values = np.linspace(float(values.min()), float(values.max()), n_values)
    dense_lags = np.linspace(float(lags.min()), float(lags.max()), n_lags)

    by_value = np.vstack([
        np.interp(dense_values, values, row)
        for row in np.asarray(rr, dtype=float)
    ])
    dense_rr = np.vstack([
        np.interp(dense_lags, lags, by_value[:, j])
        for j in range(by_value.shape[1])
    ]).T
    return dense_values, dense_lags, dense_rr


def _padded_limits(values: np.ndarray, fraction: float = 0.035) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min())
    hi = float(finite.max())
    pad = fraction * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _clean_3d_axis(
    ax,
    *,
    first: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    ax.patch.set_alpha(0.0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-48)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_box_aspect((1.0, 1.0, 0.74))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_zticks(np.round(np.linspace(zlim[0], zlim[1], 3), 2))
    ax.tick_params(axis="both", which="major", pad=-2, length=2)
    ax.tick_params(axis="z", which="major", pad=-1, length=2)
    ax.set_xlabel("Temperature", labelpad=-1)
    ax.set_ylabel("Lag", labelpad=-2)
    ax.set_zlabel("RR" if first else "", labelpad=0)   # matches OR in the malaria figure
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((0.18, 0.18, 0.18, 1))
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["linewidth"] = 0.55


def plot_comparison(payload: dict) -> list[Path]:
    curves = {model: _as_frame(payload["curves"][model]) for model in ORDER}
    surfaces = {model: _surface_matrix(payload["surfaces"][model]) for model in ORDER}
    def _zlim(values):
        finite = np.asarray(values, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.06 * max(hi - lo, 1e-9)
        return (max(0.0, lo - pad), hi + pad)

    if SURFACE_SCALE_MODE == "shared":
        shared = _zlim(np.concatenate([rr.reshape(-1) for _, _, rr in surfaces.values()]))
        zlims = {model: shared for model in surfaces}
    else:
        zlims = {model: _zlim(rr) for model, (_, _, rr) in surfaces.items()}

    value_xlim = _padded_limits(np.concatenate([curves[model]["value"].to_numpy() for model in ORDER]), 0.0)
    surface_xlims = {model: _padded_limits(values, 0.055) for model, (values, _, _) in surfaces.items()}
    surface_ylims = {model: _padded_limits(lags, 0.055) for model, (_, lags, _) in surfaces.items()}

    curve_ymin = min(float(curves[model]["lo"].min()) for model in ORDER)
    curve_ymax = max(float(curves[model]["hi"].max()) for model in ORDER)
    curve_pad = 0.04 * max(curve_ymax - curve_ymin, 1e-9)
    curve_ylim = (max(0.0, curve_ymin - curve_pad), curve_ymax + curve_pad)

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(15.2, 6.55))
        left, right, gap = 0.045, 0.99, 0.026
        width = (right - left - gap * (len(ORDER) - 1)) / len(ORDER)
        curve_y, curve_h = 0.635, 0.245
        surface_y, surface_h = 0.225, 0.430
        curve_axes = []
        surface_axes = []
        for i in range(len(ORDER)):
            x0 = left + i * (width + gap)
            curve_ax = fig.add_axes([x0, curve_y, width, curve_h])
            surface_ax = fig.add_axes([x0, surface_y, width, surface_h], projection="3d")
            curve_ax.set_zorder(5)
            curve_ax.patch.set_facecolor("white")
            curve_ax.patch.set_alpha(1.0)
            surface_ax.set_zorder(1)
            curve_axes.append(curve_ax)
            surface_axes.append(surface_ax)

        for index, (ax, model) in enumerate(zip(curve_axes, ORDER)):
            df = curves[model]
            colour = bp.COLOURS[model]
            ax.fill_between(df["value"], df["lo"], df["hi"],
                            color=colour, alpha=0.18, lw=0, zorder=1)
            ax.plot(df["value"], df["fit"], color=colour, lw=1.25, zorder=3)
            ax.axhline(1.0, color="0.88", lw=0.7, zorder=0)
            ax.set_title(bp.LABELS[model], fontsize=9, weight="bold", pad=6)
            ax.set_ylim(*curve_ylim)
            ax.set_xlim(*value_xlim)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.margins(x=0)
            ax.tick_params(axis="x", bottom=True, labelbottom=True, length=3)
            ax.spines["bottom"].set_visible(True)
            ax.set_xlabel("Temperature")
            ax.xaxis.label.set_color("0.10")
            ax.yaxis.label.set_color("0.10")
            if index == 0:
                ax.set_ylabel("Cumulative RR")
            else:
                ax.tick_params(labelleft=False)

        for index, (ax, model) in enumerate(zip(surface_axes, ORDER)):
            values, lags, rr = surfaces[model]
            plot_values, plot_lags, plot_rr = _upsample_surface(values, lags, rr)
            x, y = np.meshgrid(plot_values, plot_lags)
            colour = bp.COLOURS[model]
            surface = ax.plot_surface(
                x,
                y,
                plot_rr,
                rstride=1,
                cstride=1,
                facecolors=_surface_facecolors(plot_rr, colour),
                linewidth=0.0,
                edgecolor="none",
                antialiased=True,
                shade=False,
            )
            # ~1.4M quads as vector art is an 80 MB PDF; rasterise the surface
            # itself and leave axes, ticks and text as vectors
            surface.set_rasterized(True)
            _clean_3d_axis(
                ax,
                first=index == 0,
                xlim=surface_xlims[model],
                ylim=surface_ylims[model],
                zlim=zlims[model],
            )

        fig.suptitle("Chicago NMMAPS: Model Comparison",
                     fontsize=13, weight="bold", x=0.5, y=0.982)

        line_h = Line2D([0], [0], color="0.18", lw=1.25, label="Estimate")
        band_h = Patch(facecolor="0.35", alpha=0.18, edgecolor="none",
                       label="95% CI")
        fig.legend(
            handles=[line_h, band_h],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.155),
            fontsize=7.8,
            columnspacing=1.15,
            handletextpad=0.5,
        )

        paths = [RESULTS_DIR / f"{OUT_STEM}.png", RESULTS_DIR / f"{OUT_STEM}.pdf"]
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=450)
        plt.close(fig)
    return paths


def main() -> None:
    if "--figures-only" in sys.argv:
        if not RESULT_JSON.exists():
            raise SystemExit(f"missing {RESULT_JSON}; run this script without "
                             "--figures-only first")
        paths = plot_comparison(json.loads(RESULT_JSON.read_text(encoding="utf-8")))
        print("Chicago figure redrawn from saved results")
        for path in paths:
            print(f"  {path}")
        return

    print("Chicago NMMAPS: model comparison\n")
    df, grid, reference, n_years = prepare_chicago_bench()
    run_dlnm_fits()
    dlnam = fit_dlnam(df, grid, reference)
    dlnm_curves = {
        "QAIC": load_dlnm_curve(""),
        "QBIC": load_dlnm_curve("qbic_"),
        "Penalised": load_dlnm_curve("pen_"),
        "TDLNM": load_dlnm_curve("tree_"),
    }
    dlnm_surfaces = {
        "QAIC": load_dlnm_surface(""),
        "QBIC": load_dlnm_surface("qbic_"),
        "Penalised": load_dlnm_surface("pen_"),
        "TDLNM": load_dlnm_surface("tree_"),
    }
    payload = save_outputs(dlnam, dlnm_curves, dlnm_surfaces, grid, reference,
                           n_years)
    paths = plot_comparison(payload)
    print("\nOutputs")
    print(f"  {RESULT_JSON}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

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

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
RSCRIPT = "Rscript"
RSCRIPT_FILE = ROOT / "dlnam_bench" / "fit_chicago.R"

LAG_MAX = 30
N_GRID = 200
CI_LEVEL = 0.95
EPOCHS = 2500
N_ENSEMBLE = 3
SEED = 0
# One interval construction across every analysis: the last-layer
# Laplace variance plus the between-member ensemble spread.
SE_SOURCE = "laplace+ensemble"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# MC DLNM search grid. The penalised DLNM basis dimensions default to the upper
# end of this grid, capped by lag_max for the lag basis.
# The applied grid is narrower than the simulation's 2-10. The published
# analysis of this dataset (the dlnm package example, Gasparrini 2011) uses a
# 5-degree-of-freedom exposure margin, so this range brackets the literature
# value. Widening it to the simulation grid was tested and rejected: the three
# cross-basis fits then oscillate, QAIC placing minimum mortality at -19.7 C,
# with roughly five times the total variation of the published fit, while the
# DLNAM and the treed model are unaffected.
VALUE_DF_GRID = tuple(range(2, 6))
LAG_DF_GRID = tuple(range(2, 6))
PENALIZED_VALUE_DF = max(VALUE_DF_GRID)
PENALIZED_LAG_DF = max(LAG_DF_GRID)
# Identical to the simulation, which in turn follows Mork and Wilson (2022):
# 20 trees, 30 candidate exposure splits, 5000 burn-in, 15,000 post-burn,
# thinning by 10.
TDLNM_SETTINGS = {
    "burn": 5000,
    "iter": 15000,
    "thin": 10,
    "attempts": 10,
    "exposure_splits": 30,
    "trees": 20,
}
SURFACE_SCALE_MODE = "per_model"  # "shared" or "per_model"

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


PLOT_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.size": 8.5,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
}


def load_chicago(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["dptp", "o3", "pm10"]:
        df[f"{col}01"] = df[col].rolling(window=2).mean()
    return df


def build_dlnam_config() -> ModelConfig:
    """Chicago model with the MC surface architecture and existing adjustments."""
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    silu = lambda: ActivationSpec(base=torch.nn.SiLU)
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
    reference = float(np.median(temp))
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


def fit_dlnam(df: pd.DataFrame, grid: np.ndarray) -> dict:
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
    centering = Centering(method="median")
    extractor = EffectExtractor.with_laplace(
        trainer.ensemble, prepared, link, centering, interval=SE_SOURCE
    )
    estimate = extractor.extract("temp", grid, alpha=1.0 - CI_LEVEL)
    ref = float(trainer.ensemble[0].term("temp")._data_median)
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
                 grid: np.ndarray, reference: float, n_years: int) -> None:
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
        "tdlnm_fit_status": load_json_if_exists(BENCH_DIR / "tdlnm_fit_status.json"),
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
    save_json(RESULTS_DIR / "chicago_model_comparison.json", summary)


def _surface_matrix(df: pd.DataFrame):
    piv = df.pivot(index="lag", columns="value", values="rr").sort_index()
    vals = piv.columns.to_numpy(dtype=float)
    lags = piv.index.to_numpy(dtype=float)
    return vals, lags, piv.to_numpy(dtype=float)


def plot_comparison(dlnam: dict, dlnm_curves: dict[str, pd.DataFrame],
                    dlnm_surfaces: dict[str, pd.DataFrame]) -> list[Path]:
    curves = {"DLNAM": dlnam["curve"], **dlnm_curves}
    surfaces = {"DLNAM": dlnam["surface"], **dlnm_surfaces}
    order = ["DLNAM", "QAIC", "QBIC", "Penalised", "TDLNM"]
    n_models = len(order)
    colours = bp.COLOURS
    labels = bp.LABELS
    surface_mats = {model: _surface_matrix(surfaces[model]) for model in order}
    if SURFACE_SCALE_MODE not in {"shared", "per_model"}:
        raise ValueError("SURFACE_SCALE_MODE must be 'shared' or 'per_model'")
    if SURFACE_SCALE_MODE == "shared":
        all_rr = np.concatenate([surface_mats[model][2].reshape(-1) for model in order])
        shared_norm = bp.surface_norm(all_rr)
        surface_norms = {model: shared_norm for model in order}
    else:
        surface_norms = {
            model: bp.surface_norm(surface_mats[model][2].reshape(-1))
            for model in order
        }
    cmap = bp.surface_cmap("chicago_rr")
    curve_ymin = min(float(curves[model]["lo"].min()) for model in order)
    curve_ymax = max(float(curves[model]["hi"].max()) for model in order)
    curve_pad = 0.04 * max(curve_ymax - curve_ymin, 1e-9)
    curve_ylim = (max(0.0, curve_ymin - curve_pad), curve_ymax + curve_pad)

    with plt.rc_context(PLOT_RC):
        if SURFACE_SCALE_MODE == "per_model":
            fig = plt.figure(figsize=(15.2, 4.75))
            gs = fig.add_gridspec(
                2, n_models,
                height_ratios=[1.0, 1.05],
                wspace=0.14,
                hspace=0.18,
            )
            curve_axes = np.array([fig.add_subplot(gs[0, i]) for i in range(n_models)])
            surface_axes = np.array([
                fig.add_subplot(gs[1, i], sharex=curve_axes[i]) for i in range(n_models)
            ])
            cbar_axes = None
            cax = None
        else:
            fig = plt.figure(figsize=(15.2, 4.25))
            gs = fig.add_gridspec(
                2, n_models + 1,
                width_ratios=[1.0] * n_models + [0.035],
                height_ratios=[1.0, 1.05],
                wspace=0.14,
                hspace=0.18,
            )
            curve_axes = np.array([fig.add_subplot(gs[0, i]) for i in range(n_models)])
            surface_axes = np.array([
                fig.add_subplot(gs[1, i], sharex=curve_axes[i]) for i in range(n_models)
            ])
            cbar_axes = None
            cax = fig.add_subplot(gs[:, n_models])
        for ax, model in zip(curve_axes, order):
            df = curves[model]
            colour = colours[model]
            ax.fill_between(df["value"], df["lo"], df["hi"],
                            color=colour, alpha=0.18, lw=0, zorder=1)
            ax.plot(df["value"], df["fit"], color=colour, lw=1.25, zorder=3)
            ax.axhline(1.0, color="0.88", lw=0.7, zorder=0)
            ax.set_title(labels[model], fontsize=9, weight="bold", pad=6)
            ax.set_ylim(*curve_ylim)
            ax.margins(x=0)
            ax.tick_params(axis="x", bottom=True, labelbottom=False, length=3)
            ax.spines["bottom"].set_visible(True)
        curve_axes[0].set_ylabel("Cumulative RR", fontsize=8.5)
        for ax in curve_axes[1:]:
            ax.tick_params(labelleft=False)

        surface_image = None
        surface_images = {}
        for idx, (ax, model) in enumerate(zip(surface_axes, order)):
            vals, lags, rr = surface_mats[model]
            norm = surface_norms[model]
            surface_image = ax.imshow(
                rr,
                extent=(float(vals.min()), float(vals.max()),
                        float(lags.min()), float(lags.max())),
                origin="lower",
                aspect="auto",
                interpolation="bicubic",
                cmap=cmap,
                norm=norm,
            )
            surface_images[model] = surface_image
            ax.set_xlabel("Temperature", fontsize=8.5, labelpad=2)
            ax.set_ylim(float(lags.min()), float(lags.max()))
            ax.margins(x=0, y=0)
        surface_axes[0].set_ylabel("Lag", fontsize=8.5)
        for ax in surface_axes[1:]:
            ax.tick_params(labelleft=False)

        line_h = Line2D([0], [0], color="0.18", lw=1.25, label="Estimate")
        band_h = Patch(facecolor="0.35", alpha=0.18, edgecolor="none",
                       label="95% CI")
        if cax is not None:
            cbar = fig.colorbar(surface_image, cax=cax)
            cbar.set_label("RR", fontsize=8.5)
            cbar.ax.tick_params(labelsize=7.5, width=0.6)
            ticks = bp.surface_colorbar_ticks(surface_norms[order[-1]])
            if ticks is not None:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(bp.format_rr_ticks(ticks))
        fig.suptitle("Chicago NMMAPS: Model Comparison",
                     fontsize=13, weight="bold", x=0.5, y=0.99)
        bottom = 0.24 if SURFACE_SCALE_MODE == "per_model" else 0.15
        right = 0.975 if SURFACE_SCALE_MODE == "per_model" else 0.958
        fig.subplots_adjust(left=0.05, right=right, bottom=bottom, top=0.86)

        if SURFACE_SCALE_MODE == "per_model":
            for idx, model in enumerate(order):
                pos = surface_axes[idx].get_position()
                cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.115, pos.width, 0.018])
                cbar = fig.colorbar(surface_images[model], cax=cbar_ax,
                                    orientation="horizontal")
                cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2, pad=1)
                cbar.outline.set_linewidth(0.5)
                ticks = bp.surface_colorbar_ticks(surface_norms[model])
                if ticks is not None:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(bp.format_rr_ticks(ticks))
                if idx == 0:
                    cbar.ax.set_ylabel("RR", fontsize=7.0, rotation=0,
                                       labelpad=10, va="center")

        legend_y = 0.035 if SURFACE_SCALE_MODE == "per_model" else 0.015
        fig.legend(handles=[line_h, band_h], loc="lower center",
                   ncol=2, bbox_to_anchor=(0.5, legend_y), fontsize=8)

        paths = [
            RESULTS_DIR / "chicago_model_comparison.png",
            RESULTS_DIR / "chicago_model_comparison.pdf",
        ]
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=400)
        plt.close(fig)
    return paths


def main() -> None:
    print("Chicago NMMAPS: model comparison\n")
    df, grid, reference, n_years = prepare_chicago_bench()
    run_dlnm_fits()
    dlnam = fit_dlnam(df, grid)
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
    save_outputs(dlnam, dlnm_curves, dlnm_surfaces, grid, reference, n_years)
    paths = plot_comparison(dlnam, dlnm_curves, dlnm_surfaces)
    print("\nOutputs")
    print(f"  {RESULTS_DIR / 'chicago_model_comparison.json'}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

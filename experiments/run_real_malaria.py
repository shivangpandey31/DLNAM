"""
run_real_malaria.py -- exposure-specific malaria DLNAM vs DLNM.

The DLNM follows the thesis-era malaria specification: one focal
exposure-lag cross-basis is fitted at a time, while its prespecified adjustment
exposures enter as lag means. The corresponding DLNAM uses the same exposure-specific
adjustment set but retains a full exposure-lag surface for every adjustment
exposure.

The output is a 3x5 figure: cumulative odds ratios on the top row, DLNAM lag
surfaces in the middle row, and reference-DLNM lag surfaces on the bottom row.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Deterministic GPU reductions. Several backward kernels (notably the
# categorical embedding) accumulate with atomics, so an identical seed does not
# otherwise give an identical fit: reruns of this analysis moved minimum
# mortality by ~0.6 C. Must be set before torch initialises cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from dlnam import (
    ActivationSpec,
    Centering,
    CategoricalTermSpec,
    EffectExtractor,
    EnsembleIntervalUQ,
    ExUSpec,
    InitSpec,
    LayerSpec,
    ModelConfig,
    SmoothTermSpec,
    SurfaceTermSpec,
    TrainConfig,
    Trainer,
    make_link,
)
from dlnam.data import PreparedData
from dlnam_bench import plots as bp
from experiment_io import load_json_if_exists, results_dir, save_json


# ----------------------------- SETTINGS -----------------------------------
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

DATA_PATH = Path(os.environ.get(
    "DLNAM_MALARIA_PATH",
    ROOT / "malaria_complete.parquet",
))
BENCH_DIR = HERE / "real_malaria_data"
RESULTS_DIR = results_dir(HERE)
RESULT_JSON = RESULTS_DIR / "malaria_model_comparison.json"
RSCRIPT = "Rscript"
RSCRIPT_FILE = ROOT / "dlnam_bench" / "fit_malaria.R"

EXPOSURES = ["avg_temp", "pr", "soil", "aet", "hum"]
EXPOSURE_LABELS = {
    "avg_temp": "Temperature",
    "pr": "Precipitation",
    "soil": "Soil Moisture",
    "aet": "Evapotranspiration",
    "hum": "Humidity",
}
CONTROL_SPEC = {
    "avg_temp": ["pr"],
    "pr": ["avg_temp"],
    "soil": ["avg_temp", "pr"],
    "aet": ["soil", "avg_temp", "pr"],
    "hum": ["soil", "avg_temp", "pr", "aet"],
}
PRED_STEP = {
    "avg_temp": 0.1,
    "pr": 0.2,
    "soil": 0.1,
    "aet": 0.1,
    "hum": 0.1,
}

TARGET_COL = "test_result"
LAG_COUNT = 6
LAG_MAX_INTERNAL = LAG_COUNT - 1
# Lag enters the component as a continuous input, so the reported surface is
# evaluated between the observed monthly lags. Six lags render as ridges; this
# renders as a surface, and is a model evaluation rather than interpolation.
SURFACE_LAG_POINTS = 51   # 1.0, 1.1, ..., 6.0 -- matches bylag = 0.1 in R
N_ENSEMBLE = 3
N_SUBNETS = 3
EPOCHS = 50    # At BATCH_FRACTION = 0.01 an epoch is 100 optimisation steps,
               # so this is 5000 gradient steps.
BATCH_FRACTION = 0.01
CI_LEVEL = 0.95
SEED = 0
SUBSET_N = None
# One interval construction across every analysis: the last-layer
# Laplace variance plus the between-member ensemble spread.
SE_SOURCE = "laplace+ensemble"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFERENCE_DF_VALUE = 3
REFERENCE_DF_LAG = 3
DLNAM_INTERVAL = "laplace"
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


def _lag_cols(exposure: str) -> list[str]:
    return [f"{exposure}_lag{i}" for i in range(1, LAG_COUNT + 1)]


def load_malaria(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Malaria data not found: {path}. Set DLNAM_MALARIA_PATH to the "
            "parquet or csv file before running."
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading the malaria parquet file requires pyarrow or fastparquet. "
                "Install one of them, or provide a csv via DLNAM_MALARIA_PATH."
            ) from exc

    if SUBSET_N is not None:
        df = df.sample(n=SUBSET_N, random_state=SEED).reset_index(drop=True)

    target = df[TARGET_COL]
    if pd.api.types.is_numeric_dtype(target):
        df[TARGET_COL] = pd.to_numeric(target, errors="coerce").astype(float)
    else:
        labels = target.astype("string").str.strip().str.lower()
        y = labels.map({
            "positive": 1.0,
            "negative": 0.0,
            "pos": 1.0,
            "neg": 0.0,
            "1": 1.0,
            "0": 0.0,
            "true": 1.0,
            "false": 0.0,
        })
        bad = labels[y.isna() & labels.notna()].drop_duplicates().tolist()
        if bad:
            raise ValueError(
                f"unrecognised {TARGET_COL} values: {bad[:8]}"
            )
        df[TARGET_COL] = y.astype(float)

    for col in ["Country", "unique_cluster"]:
        df[col] = df[col].astype(str)

    required = [TARGET_COL, "month", "year", "Country", "unique_cluster"]
    for exposure in EXPOSURES:
        required.extend(_lag_cols(exposure))
    df = df.dropna(subset=required).reset_index(drop=True)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df[required].copy()


def _grid_and_reference(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    grids = {}
    refs = {}
    for exposure in EXPOSURES:
        values = df[_lag_cols(exposure)].to_numpy(dtype=float)
        lo = round(float(np.nanquantile(values, 0.025)))
        hi = round(float(np.nanquantile(values, 0.975)))
        step = float(PRED_STEP[exposure])
        ref = float(round(float(np.nanmedian(values))))
        decimals = max(0, -Decimal(str(step)).as_tuple().exponent)
        grid = np.round(
            np.arange(lo, hi + 0.5 * step, step, dtype=float), decimals
        )
        grids[exposure] = np.unique(np.append(grid, ref))
        refs[exposure] = ref
    return grids, refs


def prepare_bench(df: pd.DataFrame, grids: dict[str, np.ndarray],
                  refs: dict[str, float]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BENCH_DIR / "malaria_data.csv", index=False)
    cfg = {
        "data": "malaria_data.csv",
        "target_col": TARGET_COL,
        "exposures": EXPOSURES,
        "lag_count": LAG_COUNT,
        "ci_level": CI_LEVEL,
        "value_df": REFERENCE_DF_VALUE,
        "lag_df": REFERENCE_DF_LAG,
        "grid": {k: v.tolist() for k, v in grids.items()},
        "reference": refs,
        "control_spec": CONTROL_SPEC,
    }
    with open(BENCH_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_reference_dlnm() -> None:
    cmd = [RSCRIPT, str(RSCRIPT_FILE), str(BENCH_DIR)]
    print("\nDLNM fits")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print("  command failed: " + " ".join(cmd))
        print(result.stderr)
        raise SystemExit(f"R exited with status {result.returncode}")


def build_dlnam_config(df: pd.DataFrame,
                       active_exposures: list[str]) -> ModelConfig:
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl = lambda: InitSpec(scheme="torch_linear")

    terms = {}
    for exposure in active_exposures:
        terms[exposure] = SurfaceTermSpec(
            layers=[
                LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl()),
            ],
            num_subnets=N_SUBNETS,
            scaling="minmax",
            lag_max=LAG_MAX_INTERNAL,
            input_exu=ExUSpec(
                enabled=True,
                weight_mean=1.5,
                weight_mean_lag=2.5,
                weight_std=0.5,
                surface_strategy="concat",
                bias_init=exu_bias(),
            ),
            mix_init=mix_init(),
        )

    terms["month"] = SmoothTermSpec(
        layers=[LayerSpec(32, mish(), weight_init=tl(), bias_init=tl())],
        num_subnets=N_SUBNETS,
        scaling="zscore",
        mix_init=mix_init(),
    )
    terms["year"] = SmoothTermSpec(
        layers=[LayerSpec(32, mish(), weight_init=tl(), bias_init=tl())],
        num_subnets=N_SUBNETS,
        scaling="zscore",
        mix_init=mix_init(),
    )
    for col in ["Country", "unique_cluster"]:
        order = sorted(df[col].dropna().astype(str).unique().tolist())
        terms[col] = CategoricalTermSpec(num_categories=len(order), order=order)

    return ModelConfig(terms=terms, link="logit")


def prepare_dlnam_inputs(df: pd.DataFrame, trainer: Trainer,
                         active_exposures: list[str]) -> PreparedData:
    inputs: dict[str, torch.Tensor] = {}
    raw: dict[str, np.ndarray] = {}

    for exposure in active_exposures:
        mat = df[_lag_cols(exposure)].to_numpy(dtype=float)
        flat = mat.reshape(-1)
        for model in trainer.ensemble:
            model.term(exposure).fit_scaling(flat)
        scaled = trainer.ensemble[0].term(exposure)._to_scaled(mat)
        inputs[exposure] = torch.tensor(scaled, dtype=torch.float32)
        raw[exposure] = mat

    for col in ["month", "year"]:
        values = df[col].to_numpy(dtype=float)
        for model in trainer.ensemble:
            model.term(col).fit_scaling(values)
        scaled = trainer.ensemble[0].term(col)._to_scaled(values)
        inputs[col] = torch.tensor(scaled, dtype=torch.float32).view(-1, 1)
        raw[col] = values

    for col in ["Country", "unique_cluster"]:
        order = trainer.ensemble[0].term(col).order
        enc = {level: i for i, level in enumerate(order)}
        idx = df[col].astype(str).map(enc).to_numpy()
        if np.isnan(idx.astype(float)).any():
            raise ValueError(f"unmapped category in {col}")
        inputs[col] = torch.tensor(idx.astype(np.int64), dtype=torch.long)
        raw[col] = idx

    y = torch.tensor(df[TARGET_COL].to_numpy(dtype=float),
                     dtype=torch.float32).view(-1, 1)
    return PreparedData(inputs=inputs, y=y, raw=raw, n_samples=len(df))


def fit_dlnam(df: pd.DataFrame, grids: dict[str, np.ndarray],
              refs: dict[str, float]) -> dict:
    link = make_link("logit")
    curves = {}
    surfaces = {}
    fit_summaries = {}

    for exposure in EXPOSURES:
        active_exposures = [exposure, *CONTROL_SPEC[exposure]]
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        config = build_dlnam_config(df, active_exposures)
        train_config = TrainConfig(
            epochs=EPOCHS,
            lr=8e-4,
            lr_min=1e-4,
            weight_decay=1e-4,
            schedule="cosine",
            batch_fraction=BATCH_FRACTION,
            n_ensemble=N_ENSEMBLE,
            loss="bernoulli",
            grad_clip=10,
            seed=SEED,
        )
        trainer = Trainer(config, train_config, device=torch.device(DEVICE))
        prepared = prepare_dlnam_inputs(df, trainer, active_exposures)

        controls = ", ".join(CONTROL_SPEC[exposure]) or "none"
        print(f"\nDLNAM fit: {EXPOSURE_LABELS[exposure]}")
        print(f"  lag surfaces  {', '.join(active_exposures)}")
        print(f"  adjustments  {controls}")
        print(f"  samples       {prepared.n_samples}")
        print(f"  positive rate {float(prepared.y.mean()):.3f}")
        print(f"  reference     {refs[exposure]:g}")
        print(f"  ensemble      {N_ENSEMBLE}")
        print(f"  device        {DEVICE}")
        trainer.fit(prepared.inputs, prepared.y)

        centering = Centering(method="reference", value=refs[exposure])
        if DLNAM_INTERVAL == "laplace":
            extractor = EffectExtractor.with_laplace(
                trainer.ensemble, prepared, link, centering,
                laplace_terms=[exposure],
                interval=SE_SOURCE,
            )
        else:
            extractor = EffectExtractor(
                trainer.ensemble, link, EnsembleIntervalUQ(), centering
            )
        curve = extractor.extract(exposure, grids[exposure], alpha=1.0 - CI_LEVEL)
        surface_extractor = EffectExtractor(
            trainer.ensemble, link, EnsembleIntervalUQ(), centering
        )
        surface = surface_extractor.extract_surface(
            exposure, grids[exposure], alpha=1.0 - CI_LEVEL,
            n_lag_points=SURFACE_LAG_POINTS,
        )
        # Scaled lags run over [0, 1] across LAG_MAX_INTERNAL steps, and the
        # reported window is labelled 1..LAG_COUNT months.
        lags = np.linspace(0.0, 1.0, SURFACE_LAG_POINTS) * LAG_MAX_INTERNAL + 1.0
        curves[exposure] = pd.DataFrame({
            "value": curve.grid_raw,
            "fit": curve.mean,
            "lo": curve.lo,
            "hi": curve.hi,
        })
        surfaces[exposure] = pd.DataFrame({
            "value": np.tile(curve.grid_raw, SURFACE_LAG_POINTS),
            "lag": np.repeat(lags, len(curve.grid_raw)),
            "rr": surface.mean.reshape(-1),
        })
        fit_summaries[exposure] = trainer.fit_summary

        del extractor, surface_extractor, trainer, prepared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "curves": curves,
        "surfaces": surfaces,
        "fit_summary": fit_summaries,
    }


def load_reference_outputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    curves = {}
    surfaces = {}
    for exposure in EXPOSURES:
        curves[exposure] = pd.read_csv(BENCH_DIR / f"ref_{exposure}_cum.csv")
        surfaces[exposure] = pd.read_csv(BENCH_DIR / f"ref_{exposure}_surf.csv")
    return curves, surfaces


def save_outputs(dlnam: dict, ref_curves: dict[str, pd.DataFrame],
                 ref_surfaces: dict[str, pd.DataFrame],
                 grids: dict[str, np.ndarray], refs: dict[str, float]) -> dict:
    for exposure in EXPOSURES:
        dlnam["curves"][exposure].to_csv(
            BENCH_DIR / f"dlnam_{exposure}_cum.csv", index=False
        )
        dlnam["surfaces"][exposure].to_csv(
            BENCH_DIR / f"dlnam_{exposure}_surf.csv", index=False
        )

    summary = {
        "kind": "malaria_model_comparison",
        "models": ["DLNAM", "DLNM"],
        "exposures": EXPOSURES,
        "settings": {
            "data_path": str(DATA_PATH),
            "lag_count": LAG_COUNT,
            "epochs": EPOCHS,
            "batch_fraction": BATCH_FRACTION,
            "n_ensemble": N_ENSEMBLE,
            "n_subnets": N_SUBNETS,
            "seed": SEED,
            "device": DEVICE,
            "ci_level": CI_LEVEL,
            "dlnam_interval": DLNAM_INTERVAL,
            "reference_dlnm_value_df": REFERENCE_DF_VALUE,
            "reference_dlnm_lag_df": REFERENCE_DF_LAG,
            "reference": refs,
            "grid": {k: v.tolist() for k, v in grids.items()},
            "control_spec": CONTROL_SPEC,
            "dlnam_design": "exposure-specific prespecified adjustment",
            "dlnam_adjustment_representation": "full exposure-lag surfaces",
        },
        "dlnam_fit_summary": dlnam["fit_summary"],
        "r_environment": load_json_if_exists(BENCH_DIR / "r_environment.json"),
        "hardware": _hardware(),
        "curves": {
            exposure: {
                "DLNAM": dlnam["curves"][exposure].to_dict(orient="list"),
                "DLNM": ref_curves[exposure].to_dict(orient="list"),
            }
            for exposure in EXPOSURES
        },
        "surfaces": {
            exposure: {
                "DLNAM": dlnam["surfaces"][exposure].to_dict(orient="list"),
                "DLNM": ref_surfaces[exposure].to_dict(orient="list"),
            }
            for exposure in EXPOSURES
        },
    }
    save_json(RESULT_JSON, summary)
    return summary


OUT_STEM = "malaria_model_comparison"
MODELS = ["DLNAM", "DLNM"]
MODEL_COLOURS = {"DLNAM": bp.COLOURS["DLNAM"], "DLNM": bp.COLOURS["QBIC"]}


PLOT_RC = {
    **bp._RC,
    "axes.grid": False,
    # Matches the Chicago and DGP-surface figures so every panel in the paper
    # is typeset at the same size.
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


def _lightened(colour: str, amount: float = 0.84) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(colour), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def _surface_facecolors(rr: np.ndarray, colour: str) -> np.ndarray:
    """Single-hue surface, with height shown by lightness."""
    base = np.asarray(to_rgb(colour), dtype=float)
    light = np.asarray(_lightened(colour), dtype=float)
    z = np.asarray(rr, dtype=float)
    lo, hi = np.nanpercentile(z, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros_like(z)
    else:
        scaled = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    scaled = scaled ** 1.35
    rgb = light[None, None, :] * (1.0 - scaled[..., None]) + base[None, None, :] * scaled[..., None]
    alpha = np.full((*z.shape, 1), 0.92)
    return np.concatenate([rgb, alpha], axis=-1)


def _upsample_surface(
    values: np.ndarray,
    lags: np.ndarray,
    rr: np.ndarray,
    *,
    n_values: int = 360,
    n_lags: int = 320,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample a fitted surface for smoother 3D rendering only."""
    dense_values = np.linspace(float(values.min()), float(values.max()), n_values)
    dense_lags = np.linspace(float(lags.min()), float(lags.max()), n_lags)

    try:
        from scipy.interpolate import PchipInterpolator

        along_value = PchipInterpolator(values, rr, axis=1)(dense_values)
        dense_rr = PchipInterpolator(lags, along_value, axis=0)(dense_lags)
        try:
            from scipy.ndimage import gaussian_filter

            dense_rr = gaussian_filter(dense_rr, sigma=(3.0, 0.8), mode="nearest")
        except Exception:
            pass
        return dense_values, dense_lags, dense_rr
    except Exception:
        pass

    by_value = np.vstack([
        np.interp(dense_values, values, row)
        for row in np.asarray(rr, dtype=float)
    ])
    dense_rr = np.vstack([
        np.interp(dense_lags, lags, by_value[:, j])
        for j in range(by_value.shape[1])
    ]).T
    return dense_values, dense_lags, dense_rr


def _padded_limits(values: np.ndarray, fraction: float = 0.045) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min())
    hi = float(finite.max())
    pad = fraction * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _clean_3d_axis(
    ax,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    xlabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
    show_zlabel: bool,
) -> None:
    ax.patch.set_alpha(0.0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-48)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_box_aspect((1.0, 1.0, 0.74))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.set_zticks(np.round(np.linspace(zlim[0], zlim[1], 3), 2))
    ax.tick_params(axis="both", which="major", pad=-2, length=2)
    ax.tick_params(axis="z", which="major", pad=-1, length=2)
    ax.set_xlabel(xlabel if show_xlabel else "", labelpad=-5)
    ax.set_ylabel("Lag" if show_ylabel else "", labelpad=-6)
    ax.set_zlabel("OR" if show_zlabel else "", labelpad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((0.18, 0.18, 0.18, 1))
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["linewidth"] = 0.55


def plot_comparison(payload: dict) -> list[Path]:
    curves = {
        exposure: {
            model: _as_frame(payload["curves"][exposure][model])
            for model in MODELS
        }
        for exposure in EXPOSURES
    }
    surfaces = {
        exposure: {
            model: _surface_matrix(payload["surfaces"][exposure][model])
            for model in MODELS
        }
        for exposure in EXPOSURES
    }

    curve_ylim = {}
    surface_zlim = {}
    for exposure in EXPOSURES:
        lo = min(float(curves[exposure][m]["lo"].min()) for m in MODELS)
        hi = max(float(curves[exposure][m]["hi"].max()) for m in MODELS)
        pad = 0.05 * max(hi - lo, 1e-9)
        curve_ylim[exposure] = (max(0.0, lo - pad), hi + pad)

        pooled = np.concatenate([
            surfaces[exposure][m][2].reshape(-1)
            for m in MODELS
        ])
        zlo = float(np.nanmin(pooled))
        zhi = float(np.nanmax(pooled))
        zpad = 0.07 * max(zhi - zlo, 1e-9)
        surface_zlim[exposure] = (max(0.0, zlo - zpad), zhi + zpad)

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(15.2, 8.92))
        left, right, gap = 0.0789, 0.9912, 0.0260
        width = (right - left - gap * (len(EXPOSURES) - 1)) / len(EXPOSURES)
        curve_y, curve_h = 0.7321, 0.1799
        dlnam_y, surf_h = 0.4172, 0.3037
        dlnm_y = 0.1651

        curve_axes = []
        surface_axes = {model: [] for model in MODELS}
        for i in range(len(EXPOSURES)):
            x0 = left + i * (width + gap)
            curve_ax = fig.add_axes([x0, curve_y, width, curve_h])
            curve_ax.set_zorder(5)
            curve_ax.patch.set_facecolor("white")
            curve_ax.patch.set_alpha(1.0)
            curve_axes.append(curve_ax)
            for model, y0 in [("DLNAM", dlnam_y), ("DLNM", dlnm_y)]:
                ax3 = fig.add_axes([x0, y0, width, surf_h], projection="3d")
                ax3.set_zorder(1)
                surface_axes[model].append(ax3)

        for i, exposure in enumerate(EXPOSURES):
            ax = curve_axes[i]
            for model, z in [("DLNM", 2), ("DLNAM", 3)]:
                df = curves[exposure][model]
                colour = MODEL_COLOURS[model]
                ax.fill_between(df["value"], df["lo"], df["hi"],
                                color=colour, alpha=0.16, lw=0, zorder=z - 1)
                ax.plot(df["value"], df["fit"], color=colour, lw=1.25, zorder=z)
            ax.axhline(1.0, color="0.88", lw=0.7, zorder=0)
            ax.set_title(EXPOSURE_LABELS[exposure], fontsize=9, weight="bold", pad=6)
            ax.set_ylim(*curve_ylim[exposure])
            ax.margins(x=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="x", bottom=True, labelbottom=True, length=3)
            ax.spines["bottom"].set_visible(True)
            ax.set_xlabel(EXPOSURE_LABELS[exposure])
            if i == 0:
                ax.set_ylabel("Cumulative OR")

            for model in MODELS:
                values, lags, rr = surfaces[exposure][model]
                plot_values, plot_lags, plot_rr = _upsample_surface(values, lags, rr)
                x_mesh, lag_mesh = np.meshgrid(plot_values, plot_lags)
                ax3 = surface_axes[model][i]
                surface = ax3.plot_surface(
                    x_mesh,
                    lag_mesh,
                    plot_rr,
                    rstride=1,
                    cstride=1,
                    facecolors=_surface_facecolors(plot_rr, MODEL_COLOURS[model]),
                    linewidth=0.0,
                    edgecolor="none",
                    antialiased=False,
                    shade=False,
                )
                surface.set_rasterized(True)
                _clean_3d_axis(
                    ax3,
                    xlim=_padded_limits(values, 0.045),
                    ylim=_padded_limits(lags, 0.045),
                    zlim=surface_zlim[exposure],
                    xlabel=EXPOSURE_LABELS[exposure],
                    show_xlabel=True,
                    show_ylabel=True,
                    show_zlabel=i == 0,
                )

        curve_axes[0].text(
            -0.28,
            0.5,
            "Comparison",
            transform=curve_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )
        surface_axes["DLNAM"][0].text2D(
            -0.28,
            0.55,
            "DLNAM",
            transform=surface_axes["DLNAM"][0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )
        surface_axes["DLNM"][0].text2D(
            -0.28,
            0.55,
            "DLNM",
            transform=surface_axes["DLNM"][0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )

        fig.suptitle("DHS/MIS Malaria: Model Comparison",
                     fontsize=13, weight="bold", x=0.5351, y=0.9868)

        line_d = Line2D([0], [0], color=MODEL_COLOURS["DLNAM"], lw=1.25, label="DLNAM")
        line_r = Line2D([0], [0], color=MODEL_COLOURS["DLNM"], lw=1.25, label="DLNM")
        band_h = Patch(facecolor="0.35", alpha=0.18, edgecolor="none",
                       label="95% CI")
        fig.legend(
            handles=[line_d, line_r, band_h],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5351, 0.1137),
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
        print("DHS/MIS Malaria figure redrawn from saved results")
        for path in paths:
            print(f"  {path}")
        return

    print("Malaria data: exposure-specific lag model comparison\n")
    df = load_malaria(DATA_PATH)
    grids, refs = _grid_and_reference(df)
    print("Data")
    print(f"  observations   {len(df)}")
    print(f"  positive rate  {df[TARGET_COL].mean():.3f}")
    print(f"  exposures      {', '.join(EXPOSURES)}")
    print(f"  lag count      {LAG_COUNT}")

    prepare_bench(df, grids, refs)
    run_reference_dlnm()
    dlnam = fit_dlnam(df, grids, refs)
    ref_curves, ref_surfaces = load_reference_outputs()
    payload = save_outputs(dlnam, ref_curves, ref_surfaces, grids, refs)
    paths = plot_comparison(payload)

    print("\nOutputs")
    print(f"  {RESULT_JSON}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

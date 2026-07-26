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

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
N_ENSEMBLE = 3
N_SUBNETS = 3
EPOCHS = 25
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


PLOT_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.size": 8.2,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
}


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
            exposure, grids[exposure], alpha=1.0 - CI_LEVEL
        )
        lags = np.arange(1, LAG_COUNT + 1, dtype=float)
        curves[exposure] = pd.DataFrame({
            "value": curve.grid_raw,
            "fit": curve.mean,
            "lo": curve.lo,
            "hi": curve.hi,
        })
        surfaces[exposure] = pd.DataFrame({
            "value": np.tile(curve.grid_raw, LAG_COUNT),
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


def _surface_matrix(df: pd.DataFrame):
    surface = df.loc[:, ["lag", "value", "rr"]].copy()
    surface[["lag", "value"]] = surface[["lag", "value"]].round(12)
    coordinates = ["lag", "value"]
    if surface.duplicated(coordinates).any():
        grouped = surface.groupby(coordinates, sort=False)["rr"]
        spread = grouped.agg(lambda values: values.max() - values.min())
        tolerance = 1e-10 * max(1.0, float(surface["rr"].abs().max()))
        if float(spread.max()) > tolerance:
            raise ValueError(
                "surface contains conflicting predictions at one coordinate"
            )
        surface = grouped.mean().reset_index()
    piv = surface.pivot(
        index="lag", columns="value", values="rr"
    ).sort_index()
    vals = piv.columns.to_numpy(dtype=float)
    lags = piv.index.to_numpy(dtype=float)
    return vals, lags, piv.to_numpy(dtype=float)


def save_outputs(dlnam: dict, ref_curves: dict[str, pd.DataFrame],
                 ref_surfaces: dict[str, pd.DataFrame],
                 grids: dict[str, np.ndarray], refs: dict[str, float]) -> None:
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
    save_json(RESULTS_DIR / "malaria_model_comparison.json", summary)


def plot_comparison(dlnam: dict, ref_curves: dict[str, pd.DataFrame],
                    ref_surfaces: dict[str, pd.DataFrame]) -> list[Path]:
    dlnam_colour = bp.COLOURS["DLNAM"]
    ref_colour = bp.COLOURS["QBIC"]
    cmap = bp.surface_cmap("malaria_or")
    n_exp = len(EXPOSURES)

    surface_mats = {}
    norms = {}
    for exposure in EXPOSURES:
        dm = _surface_matrix(dlnam["surfaces"][exposure])
        rm = _surface_matrix(ref_surfaces[exposure])
        surface_mats[exposure] = {"DLNAM": dm, "DLNM": rm}
        pooled = np.concatenate([dm[2].reshape(-1), rm[2].reshape(-1)])
        norms[exposure] = bp.surface_norm(pooled)

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(15.2, 7.0))
        gs = fig.add_gridspec(
            3, n_exp,
            height_ratios=[0.95, 1.0, 1.0],
            wspace=0.18,
            hspace=0.22,
        )
        curve_axes = [fig.add_subplot(gs[0, i]) for i in range(n_exp)]
        dlnam_axes = [fig.add_subplot(gs[1, i], sharex=curve_axes[i])
                      for i in range(n_exp)]
        ref_axes = [fig.add_subplot(gs[2, i], sharex=curve_axes[i])
                    for i in range(n_exp)]

        images = {}
        for i, exposure in enumerate(EXPOSURES):
            ax = curve_axes[i]
            dc = dlnam["curves"][exposure]
            rc = ref_curves[exposure]
            for df, colour, z in [(rc, ref_colour, 2), (dc, dlnam_colour, 3)]:
                ax.fill_between(df["value"], df["lo"], df["hi"],
                                color=colour, alpha=0.14, lw=0, zorder=z - 1)
                ax.plot(df["value"], df["fit"], color=colour, lw=1.25, zorder=z)
            ax.axhline(1.0, color="0.88", lw=0.7, zorder=0)
            ax.set_title(EXPOSURE_LABELS[exposure], fontsize=9, weight="bold", pad=6)
            ax.tick_params(axis="x", bottom=True, labelbottom=False, length=3)
            ax.spines["bottom"].set_visible(True)
            ax.margins(x=0)
            ax.set_ylim(bottom=0.0)
            if i == 0:
                ax.set_ylabel("Cumulative OR", fontsize=8.5)

            for row_ax, model in [(dlnam_axes[i], "DLNAM"),
                                  (ref_axes[i], "DLNM")]:
                vals, lags, rr = surface_mats[exposure][model]
                im = row_ax.imshow(
                    rr,
                    extent=(float(vals.min()), float(vals.max()),
                            float(lags.min()), float(lags.max())),
                    origin="lower",
                    aspect="auto",
                    interpolation="bicubic",
                    cmap=cmap,
                    norm=norms[exposure],
                )
                images[exposure] = im
                row_ax.set_ylim(float(lags.min()), float(lags.max()))
                row_ax.margins(x=0, y=0)
                if i == 0:
                    row_ax.set_ylabel("Lag", fontsize=8.5)
                else:
                    row_ax.tick_params(labelleft=False)
            dlnam_axes[i].tick_params(axis="x", bottom=True, labelbottom=False, length=3)
            dlnam_axes[i].spines["bottom"].set_visible(True)
            ref_axes[i].set_xlabel(EXPOSURE_LABELS[exposure], fontsize=8.5, labelpad=2)

        curve_axes[0].text(-0.28, 0.5, "Comparison", transform=curve_axes[0].transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=9, weight="bold")
        dlnam_axes[0].text(-0.28, 0.5, "DLNAM", transform=dlnam_axes[0].transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=9, weight="bold")
        ref_axes[0].text(-0.28, 0.5, "DLNM", transform=ref_axes[0].transAxes,
                         rotation=90, ha="center", va="center",
                         fontsize=9, weight="bold")

        fig.suptitle("Malaria Data: Model Comparison",
                     fontsize=13, weight="bold", x=0.5, y=0.985)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.24, top=0.90)

        for i, exposure in enumerate(EXPOSURES):
            pos = ref_axes[i].get_position()
            cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.085, pos.width, 0.018])
            cbar = fig.colorbar(images[exposure], cax=cbar_ax,
                                orientation="horizontal")
            cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2, pad=1)
            cbar.outline.set_linewidth(0.5)
            ticks = bp.surface_colorbar_ticks(norms[exposure])
            if ticks is not None:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(bp.format_rr_ticks(ticks))
            if i == 0:
                cbar.ax.set_ylabel("OR", fontsize=7.0, rotation=0,
                                   labelpad=10, va="center")

        line_d = Line2D([0], [0], color=dlnam_colour, lw=1.25, label="DLNAM")
        line_r = Line2D([0], [0], color=ref_colour, lw=1.25,
                        label="DLNM")
        band_h = Patch(facecolor="0.35", alpha=0.18, edgecolor="none",
                       label="95% CI")
        fig.legend(handles=[line_d, line_r, band_h],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.08),
                   fontsize=8)

        paths = [
            RESULTS_DIR / "malaria_model_comparison.png",
            RESULTS_DIR / "malaria_model_comparison.pdf",
        ]
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=400)
        plt.close(fig)
    return paths


def main() -> None:
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
    save_outputs(dlnam, ref_curves, ref_surfaces, grids, refs)
    paths = plot_comparison(dlnam, ref_curves, ref_surfaces)

    print("\nOutputs")
    print(f"  {RESULTS_DIR / 'malaria_model_comparison.json'}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

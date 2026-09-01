from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlnam_bench import plots as bp
from experiment_io import collect_environment

# Typography and line/marker weights mirror the Monte Carlo figure family.

# Palette, markers and labels come from the shared figure module so that each
# estimator is drawn identically in every figure.
MODEL_ORDER = list(bp.MODELS)
LABELS = bp.LABELS
COLOURS = bp.COLOURS
MARKERS = bp.MARKERS

MC_RC = {
    **bp._RC,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.8,
}


class WhiskerHandle:
    """Dummy handle used only for a min-max whisker legend entry."""


# The min-max whisker glyph is shared with the Monte Carlo figures so the legend
# symbol matches them exactly.
WhiskerHandler = bp._WhiskerHandler


def _relative_summary(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def add_runtime_relative(results: dict, exposure_counts: tuple[int, ...]) -> None:
    """Recompute within-model runtime ratios against M=1 from raw repeats."""
    base_key = str(exposure_counts[0])
    for model in MODEL_ORDER:
        base = results.get(base_key, {}).get(model, {}).get("runtime", {}).get("seconds", [])
        if not base:
            continue
        base_values = np.asarray(base, dtype=float)
        base_median = float(np.median(base_values))
        for count in exposure_counts:
            entry = results.get(str(count), {}).get(model)
            if not entry:
                continue
            values = entry.get("runtime", {}).get("seconds", [])
            if not values:
                continue
            values = np.asarray(values, dtype=float)
            relative = values / (base_values if len(values) == len(base_values) else base_median)
            entry["runtime_relative"] = _relative_summary(relative)


def _plain_log_tick(value, _pos):
    if value <= 0:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def render(input_json: Path, out_pdf: Path, out_png: Path) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    results = payload["results"]
    exposure_counts = tuple(int(x) for x in payload["settings"]["exposure_counts"])
    add_runtime_relative(results, exposure_counts)
    models = [
        model
        for model in payload["models"]
        if any(model in results.get(str(count), {}) for count in exposure_counts)
    ]

    with plt.rc_context(MC_RC):
        # Keep the existing wide two-panel geometry; LaTeX places this at 0.70\linewidth.
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
        axes[0].set_title("Absolute", fontsize=9, fontweight="bold", pad=6)
        axes[1].set_title("Relative", fontsize=9, fontweight="bold", pad=6)

        for model in models:
            style = dict(
                color=COLOURS[model],
                marker=MARKERS[model],
                markersize=4.8,
                linewidth=1.05,
                elinewidth=0.75,
                capsize=2.0,
                zorder=5,
            )

            # Absolute runtime (minutes).
            x, y, lo, hi = [], [], [], []
            for count in exposure_counts:
                summary = results.get(str(count), {}).get(model, {}).get("runtime", {})
                if not summary.get("n"):
                    continue
                med = summary["median_seconds"] / 60.0
                low = summary["min_seconds"] / 60.0
                high = summary["max_seconds"] / 60.0
                x.append(count)
                y.append(med)
                lo.append(med - low)
                hi.append(high - med)
            if x:
                axes[0].errorbar(x, y, yerr=np.asarray([lo, hi]), label="_nolegend_", **style)

            # Runtime relative to each estimator's one-exposure fit.
            x, y, lo, hi = [], [], [], []
            for count in exposure_counts:
                summary = results.get(str(count), {}).get(model, {}).get("runtime_relative", {})
                if not summary.get("n"):
                    continue
                med = summary["median"]
                low = summary["min"]
                high = summary["max"]
                x.append(count)
                y.append(med)
                lo.append(med - low)
                hi.append(high - med)
            if x:
                axes[1].errorbar(x, y, yerr=np.asarray([lo, hi]), label="_nolegend_", **style)

        axes[0].set_ylabel("Runtime (min)")
        axes[1].set_ylabel("Relative Runtime")
        for ax in axes:
            ax.set_xlabel("Number of Exposures")
            ax.set_xticks(exposure_counts)
            ax.tick_params(direction="out", length=3)
            ax.set_yscale("log")
            ax.grid(False)
            ax.set_xlim(min(exposure_counts) - 0.15, max(exposure_counts) + 0.15)

        all_abs = [
            results[str(c)][m]["runtime"]["max_seconds"] / 60.0
            for c in exposure_counts
            for m in models
            if m in results.get(str(c), {})
        ]
        all_rel = [
            results[str(c)][m]["runtime_relative"]["max"]
            for c in exposure_counts
            for m in models
            if m in results.get(str(c), {})
        ]
        axes[0].set_ylim(max(min(all_abs) * 0.75, 0.03), max(all_abs) * 1.35)
        axes[1].set_ylim(0.85, max(all_rel) * 1.35)

        axes[0].yaxis.set_major_locator(LogLocator(base=10))
        axes[1].yaxis.set_major_locator(LogLocator(base=2))
        axes[0].yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))
        axes[1].yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))
        axes[0].yaxis.set_minor_formatter(NullFormatter())
        axes[1].yaxis.set_minor_formatter(NullFormatter())

        fig.suptitle(
            "Computational Scaling: Model Comparison",
            fontsize=13,
            fontweight="bold",
            y=0.975,
        )

        model_handles = [
            Line2D(
                [0], [0],
                marker=MARKERS[model],
                color=COLOURS[model],
                linewidth=0,
                markersize=6,
                label=LABELS[model],
            )
            for model in models
        ]
        whisker = WhiskerHandle()
        fig.legend(
            handles=model_handles + [whisker],
            labels=[LABELS[m] for m in models] + ["Min-Max"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.002),
            ncol=len(model_handles) + 1,
            fontsize=7.8,
            columnspacing=1.05,
            handletextpad=0.5,
            frameon=False,
            handler_map={WhiskerHandle: WhiskerHandler(), whisker: WhiskerHandler()},
        )

        fig.subplots_adjust(
            left=0.095,
            right=0.985,
            top=0.79,
            bottom=0.19,
            wspace=0.31,
        )
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmark runner
#
# Measures end-to-end fit time as concurrently lagged exposures are added, at a
# fixed DLNAM training budget. The timed scope is model construction,
# preprocessing, fitting, uncertainty estimation, and cumulative-effect
# extraction; data generation and process startup are excluded. Exposures are
# added in the order dgp1, dgp2, dgp3, dgp4 on the joint simulation design.
#
# Absolute timings are hardware-dependent, so re-running reproduces the
# procedure rather than the numbers recorded in the shipped benchmark.
# ---------------------------------------------------------------------------

BENCH_EXPOSURE_COUNTS = (1, 2, 3, 4)
TIMING_REPEATS = 3
BENCH_N_OBS = 5000
BENCH_LAG = 14
BENCH_EPOCHS = 2500
BENCH_N_ENSEMBLE = 3
BENCH_N_SUBNETS = 3
BENCH_SEED = 0
BENCH_LR = 8e-4
BENCH_LR_MIN = 1e-4
BENCH_WEIGHT_DECAY = 1e-4
BENCH_GRAD_CLIP = 10.0
BENCH_ALPHA = 0.05
VALUE_DF_GRID = tuple(range(2, 11))
LAG_DF_GRID = tuple(range(2, 11))
PENALIZED_VALUE_DF = 10
PENALIZED_LAG_DF = 10
TDLNM_SETTINGS = {
    "burn": 5000, "iter": 15000, "thin": 10, "attempts": 10,
    "exposure_splits": 30, "trees": 20, "adjust_value_df": 4, "adjust_lag_df": 4,
}
R_METHOD_MODEL = {"qaic": "QAIC", "qbic": "QBIC", "pen": "Penalised", "tdlnm": "TDLNM"}
RSCRIPT_ENV_KEY = "r_environment"
_LAST_R_ENVIRONMENT = None
RSCRIPT = "Rscript"


def _hardware() -> dict:
    import platform
    import torch
    hw = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw


def _bench_dgp(names):
    """Joint DGP restricted to the first len(names) exposures.

    scenarios_joint builds its exposure set from a module constant, so the
    constant is set for the duration of the call rather than duplicating the
    correlated-exposure construction here.
    """
    from dlnam_sim import scenarios_joint as MX
    original = MX.EXPOSURES
    try:
        MX.EXPOSURES = tuple(names)
        return MX.joint_dgp(lag_max=BENCH_LAG), float(MX.REFERENCE)
    finally:
        MX.EXPOSURES = original


def _export_case(bench_dir, dgp, names, seed, grid, reference):
    """One dataset plus the manifest fit_joint_mc.R expects."""
    import pandas as pd
    os.makedirs(os.path.join(bench_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(bench_dir, "out"), exist_ok=True)
    sim = dgp.simulate(BENCH_N_OBS, seed)
    cols = {name: sim.frame[name].values for name in names}
    cols[dgp.target_col] = sim.frame[dgp.target_col].values
    rel = "data/bench_rep000.csv"
    pd.DataFrame(cols).to_csv(os.path.join(bench_dir, rel), index=False)
    manifest = {
        "target_col": dgp.target_col,
        "exposures": list(names),
        "lag_max": int(BENCH_LAG),
        "reference": float(reference),
        "alpha": float(BENCH_ALPHA),
        "ci_level": float(1 - BENCH_ALPHA),
        "grid": [float(v) for v in np.asarray(grid)],
        "value_df_grid": list(VALUE_DF_GRID),
        "lag_df_grid": list(LAG_DF_GRID),
        "penalized_value_df": int(PENALIZED_VALUE_DF),
        "penalized_lag_df": int(PENALIZED_LAG_DF),
        "n_obs": int(BENCH_N_OBS),
        "n_reps": 1,
        "base_seed": int(seed),
        "datasets": [{
            "rep": 0, "seed": int(seed), "data": rel,
            "cumulative": {n: f"out/bench_rep000_{n}_cum.csv" for n in names},
            "surface": {n: f"out/bench_rep000_{n}_surf.csv" for n in names},
        }],
    }
    for k, v in TDLNM_SETTINGS.items():
        manifest["tdlnm_" + k] = int(v)
    with open(os.path.join(bench_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return sim


def _time_dlnam(sim, dgp, names, grid, reference):
    """Time the reported scope for one DLNAM fit; returns (seconds, extras)."""
    import time
    import torch
    from dlnam import (Centering, DataProcessor, EffectExtractor, Trainer,
                       TrainConfig, make_link)
    from dlnam.config import (ActivationSpec, ExUSpec, InitSpec, LayerSpec,
                              ModelConfig, SurfaceTermSpec)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(BENCH_SEED)
    np.random.seed(BENCH_SEED)

    start = time.perf_counter()
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    tl = lambda: InitSpec(scheme="torch_linear")
    terms = {
        name: SurfaceTermSpec(
            layers=[LayerSpec(128, mish()),
                    LayerSpec(128, mish(), weight_init=tl(), bias_init=tl())],
            num_subnets=BENCH_N_SUBNETS,
            scaling="minmax",
            lag_max=BENCH_LAG,
            input_exu=ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                              weight_std=0.5, surface_strategy="concat",
                              bias_init=InitSpec(scheme="uniform", lo=0.0, hi=1.0)),
            mix_init=InitSpec(scheme="normal", mean=0.0, std=0.1),
        )
        for name in names
    }
    config = ModelConfig(terms=terms, link="log")
    tcfg = TrainConfig(epochs=BENCH_EPOCHS, n_ensemble=BENCH_N_ENSEMBLE,
                       lr=BENCH_LR, lr_min=BENCH_LR_MIN,
                       weight_decay=BENCH_WEIGHT_DECAY, schedule="cosine",
                       grad_clip=BENCH_GRAD_CLIP, seed=BENCH_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, tcfg, device=device)
    prepared = DataProcessor(config).prepare(sim.frame, trainer.ensemble,
                                             target_col=dgp.target_col)
    fit_start = time.perf_counter()
    trainer.fit(prepared.inputs, prepared.y)
    training_seconds = time.perf_counter() - fit_start

    extractor = EffectExtractor.with_laplace(
        trainer.ensemble, prepared, make_link("log"),
        Centering(method="reference", value=reference),
        interval="laplace+ensemble")
    for name in names:
        extractor.extract(name, grid, alpha=BENCH_ALPHA)
    seconds = time.perf_counter() - start

    extras = {"training_seconds": float(training_seconds)}
    if torch.cuda.is_available():
        extras["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        extras["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    try:
        import psutil
        extras["peak_host_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    return seconds, extras


def _run_r(bench_dir):
    """Fit the four DLNM-family comparators; returns per-method seconds."""
    import subprocess
    import time
    rscript = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dlnam_bench", "fit_joint_mc.R")
    methods = ",".join(R_METHOD_MODEL)
    start = time.perf_counter()
    proc = subprocess.run([RSCRIPT, rscript, bench_dir, methods],
                          capture_output=True, text=True)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"R exited with status {proc.returncode}")
    timing_path = os.path.join(bench_dir, "timing.json")
    if not os.path.exists(timing_path):
        raise SystemExit(f"R produced no timing record at {timing_path}")
    global _LAST_R_ENVIRONMENT
    env_path = os.path.join(bench_dir, "r_environment.json")
    if os.path.exists(env_path):
        _LAST_R_ENVIRONMENT = json.load(open(env_path))
    timing = json.load(open(timing_path))
    records = timing.get("records", timing if isinstance(timing, list) else [])
    per_method = {}
    for rec in records:
        model = R_METHOD_MODEL.get(str(rec.get("method", "")).lower())
        if model is None:
            continue
        per_method.setdefault(model, 0.0)
        per_method[model] += float(rec.get("fit_seconds", 0.0))
    return per_method, wall


def run_benchmark(exposure_counts=BENCH_EXPOSURE_COUNTS, repeats=TIMING_REPEATS,
                  epochs=BENCH_EPOCHS, bench_dir=None):
    """Run the scaling benchmark and return a record in the published schema."""
    import tempfile
    from datetime import datetime, timezone
    from dlnam_sim import scenarios_joint as MX

    global BENCH_EPOCHS
    BENCH_EPOCHS = int(epochs)

    order = list(MX.EXPOSURES)
    tmp = bench_dir or tempfile.mkdtemp(prefix="dlnam_bench_")
    results, details = {}, {}

    for m in exposure_counts:
        names = order[:m]
        dgp, reference = _bench_dgp(names)
        grid = np.asarray(dgp.terms[names[0]].default_grid(), dtype=float) \
            if hasattr(dgp, "terms") else np.linspace(10.0, 30.0, 41)
        results[str(m)], details[str(m)] = {}, {}
        for rep in range(repeats):
            seed = BENCH_SEED + rep
            sim = _export_case(tmp, dgp, names, seed, grid, reference)
            print(f"  M={m} rep={rep}: R comparators", flush=True)
            r_seconds, _ = _run_r(tmp)
            print(f"  M={m} rep={rep}: DLNAM", flush=True)
            dl_seconds, extras = _time_dlnam(sim, dgp, names, grid, reference)
            row = {"DLNAM": dl_seconds}
            row.update(r_seconds)
            for model, secs in row.items():
                details[str(m)].setdefault(model, []).append(
                    dict(repeat=rep, seed=seed, seconds=float(secs),
                         **(extras if model == "DLNAM" else {})))
        for model, recs in details[str(m)].items():
            vals = np.asarray([r["seconds"] for r in recs], dtype=float)
            results[str(m)][model] = {"runtime": {
                "seconds": [float(v) for v in vals], "n": int(vals.size),
                "median_seconds": float(np.median(vals)),
                "min_seconds": float(vals.min()),
                "max_seconds": float(vals.max())}}

    return {
        "kind": "runtime_scaling",
        "settings": {
            "exposure_counts": list(exposure_counts), "timing_repeats": int(repeats),
            "n_obs": BENCH_N_OBS, "lag": BENCH_LAG, "epochs": int(epochs),
            "n_ensemble": BENCH_N_ENSEMBLE, "n_subnets": BENCH_N_SUBNETS,
            "reference": 20.0, "seed": BENCH_SEED,
            "exu_weight_mean": 1.5, "exu_lag_weight_mean": 2.5, "exu_weight_std": 0.5,
            "learning_rate": BENCH_LR, "min_learning_rate": BENCH_LR_MIN,
            "weight_decay": BENCH_WEIGHT_DECAY, "grad_clip": BENCH_GRAD_CLIP,
            "value_df_grid": list(VALUE_DF_GRID), "lag_df_grid": list(LAG_DF_GRID),
            "penalized_value_df": PENALIZED_VALUE_DF,
            "penalized_lag_df": PENALIZED_LAG_DF,
            "tdlnm": dict(TDLNM_SETTINGS),
            "timer_scope": ("model construction, preprocessing, fitting, uncertainty "
                            "estimation, and cumulative-effect extraction"),
            "memory_scope": ("peak allocated CUDA memory for DLNAM and end-of-fit "
                             "resident memory of the running process"),
            "data_generation_timed": False, "process_startup_timed": False,
        },
        "models": ["DLNAM", "QAIC", "QBIC", "Penalised", "TDLNM"],
        "exposure_order": order,
        "hardware": _hardware(),
        "results": results,
        "details": details,
        "r_environment": _LAST_R_ENVIRONMENT,
        "environment": collect_environment(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the computational-scaling benchmark and draw its figure.")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw from a stored benchmark record without refitting")
    parser.add_argument("--input", type=Path, default=None,
                        help="benchmark JSON (default: experiments/results/runtime_scaling.json)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: experiments/results)")
    parser.add_argument("--exposure-counts", type=int, nargs="+",
                        default=list(BENCH_EXPOSURE_COUNTS),
                        help="numbers of concurrent exposures to time")
    parser.add_argument("--repeats", type=int, default=TIMING_REPEATS,
                        help="timed repeats per method and exposure count")
    parser.add_argument("--epochs", type=int, default=BENCH_EPOCHS,
                        help="DLNAM training budget held fixed across exposure counts")
    args = parser.parse_args()

    odir = args.outdir or _results_dir()
    odir.mkdir(parents=True, exist_ok=True)
    src = args.input or (odir / "runtime_scaling.json")

    if not args.figures_only:
        record = run_benchmark(exposure_counts=tuple(args.exposure_counts),
                               repeats=args.repeats, epochs=args.epochs)
        with open(src, "w") as f:
            json.dump(record, f, indent=2)
        print(f"wrote {src}")
    elif not src.exists():
        raise SystemExit(
            f"missing benchmark record: {src}\n"
            "Run without --figures-only to produce it.")

    render(src, odir / "runtime_scaling.pdf", odir / "runtime_scaling.png")
    print(f"wrote {odir / 'runtime_scaling.pdf'}")
    print(f"wrote {odir / 'runtime_scaling.png'}")

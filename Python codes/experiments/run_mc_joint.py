"""
run_mc_joint.py -- paired MC comparison for one joint four-exposure DGP.

This is the joint-exposure analogue of run_mc.py. It fits one joint DLNAM with
one surface term per exposure and compares it with joint DLNM-family fits in R.
TDLNM is fitted target-exposure by target-exposure, with the other concurrent
exposures included as fixed cross-basis adjustment terms.
The scoring contract is intentionally identical to the main MC: cumulative RR
curves, logRR RMSE, bias^2, variance and pointwise coverage, split into total,
interior and boundary regions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from dlnam.config import (
    ActivationSpec,
    ExUSpec,
    InitSpec,
    LayerSpec,
    ModelConfig,
    SurfaceTermSpec,
    TrainConfig,
)
from dlnam.terms.base import Centering
from dlnam_sim.study import MonteCarloStudy, ReplicateResult, StudyResult

from dlnam_bench import plots as bp
from dlnam_sim import scenarios_joint as MX
from experiment_io import load_json_if_exists, results_dir, save_result_bundle


# ----------------------------- SETTINGS -----------------------------------
LAG = 14
N_REPS = 3
N_OBS = 5000
EPOCHS = 2500
N_ENSEMBLE = 3
REF = MX.REFERENCE
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_joint_data")
RSCRIPT = "Rscript"
VALUE_DF_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
LAG_DF_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
PENALIZED_VALUE_DF = max(VALUE_DF_GRID)
PENALIZED_LAG_DF = max(LAG_DF_GRID)
TDLNM_SETTINGS = {
    "burn": 5000,
    "iter": 15000,
    "thin": 5,
    "attempts": 3,
    "exposure_splits": 20,
    "adjust_value_df": 4,
    "adjust_lag_df": 4,
}
R_MODELS = {
    "QAIC": {"method": "qaic", "prefix": "qaic_"},
    "QBIC": {"method": "qbic", "prefix": "qbic_"},
    "Penalised": {"method": "pen", "prefix": "pen_"},
    "TDLNM": {"method": "tdlnm", "prefix": "tree_"},
}
R_METHODS = ",".join(spec["method"] for spec in R_MODELS.values())
INCLUDE_NULL = True
EXPOSURES = list(MX.EXPOSURES) + (["null"] if INCLUDE_NULL else [])
PLOT_EXPOSURES = [name for name in EXPOSURES if name != "null"]
# --------------------------------------------------------------------------


def surface_spec(lag):
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl = lambda: InitSpec(scheme="torch_linear")
    return SurfaceTermSpec(
        layers=[
            LayerSpec(128, mish()),
            LayerSpec(128, mish(), weight_init=tl(), bias_init=tl()),
        ],
        num_subnets=N_ENSEMBLE,
        scaling="minmax",
        lag_max=lag,
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


def dlnam_config(lag):
    return ModelConfig(terms={name: surface_spec(lag) for name in EXPOSURES}, link="log")


def summarise_term(study, term, mask_int=None, mask_bnd=None):
    regions = [("tot", None), ("int", mask_int), ("bnd", mask_bnd)]
    out = {}
    for tag, m in regions:
        out[f"err_{tag}"], out[f"err_{tag}_se"] = study.rmse_mean_se(term, mask=m)
        out[f"bias2_{tag}"], out[f"bias2_{tag}_se"] = study.bias2_mean_se(term, mask=m)
        out[f"var_{tag}"], out[f"var_{tag}_se"] = study.variance_mean_se(term, mask=m)
        out[f"cov_{tag}"], out[f"cov_{tag}_se"] = study.coverage_mean_se(term, mask=m)
    return out


def export_joint_datasets(
    dgp,
    out_dir,
    *,
    n_reps,
    n_obs,
    grid,
    reference,
    base_seed=0,
    alpha=0.05,
):
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)

    datasets = []
    for rep in range(n_reps):
        seed = base_seed + rep
        sim = dgp.simulate(n_obs, seed)
        cols = {name: sim.frame[name].values for name in EXPOSURES}
        cols[dgp.target_col] = sim.frame[dgp.target_col].values
        rel = f"data/joint_rep{rep:03d}.csv"
        pd.DataFrame(cols).to_csv(os.path.join(out_dir, rel), index=False)
        datasets.append({
            "rep": int(rep),
            "seed": int(seed),
            "data": rel,
            "cumulative": {
                name: f"out/joint_rep{rep:03d}_{name}_cum.csv" for name in EXPOSURES
            },
            "surface": {
                name: f"out/joint_rep{rep:03d}_{name}_surf.csv" for name in EXPOSURES
            },
        })

    manifest = {
        "target_col": dgp.target_col,
        "exposures": EXPOSURES,
        "lag_max": int(LAG),
        "reference": float(reference),
        "alpha": float(alpha),
        "ci_level": float(1 - alpha),
        "grid": [float(v) for v in np.asarray(grid)],
        "value_df_grid": list(VALUE_DF_GRID),
        "lag_df_grid": list(LAG_DF_GRID),
        "penalized_value_df": int(PENALIZED_VALUE_DF),
        "penalized_lag_df": int(PENALIZED_LAG_DF),
        "tdlnm_burn": int(TDLNM_SETTINGS["burn"]),
        "tdlnm_iter": int(TDLNM_SETTINGS["iter"]),
        "tdlnm_thin": int(TDLNM_SETTINGS["thin"]),
        "tdlnm_attempts": int(TDLNM_SETTINGS["attempts"]),
        "tdlnm_exposure_splits": int(TDLNM_SETTINGS["exposure_splits"]),
        "tdlnm_adjust_value_df": int(TDLNM_SETTINGS["adjust_value_df"]),
        "tdlnm_adjust_lag_df": int(TDLNM_SETTINGS["adjust_lag_df"]),
        "n_obs": int(n_obs),
        "n_reps": int(n_reps),
        "base_seed": int(base_seed),
        "effect_scale": float(MX.EFFECT_SCALE),
        "exposure_corr": float(MX.EXPOSURE_CORR),
        "datasets": datasets,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return os.path.join(out_dir, "manifest.json")


def _prefixed(path, prefix):
    d, b = os.path.split(path)
    return os.path.join(d, prefix + b)


def load_joint_dlnm_study(out_dir, dgp, centering, prefix=""):
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    grid = np.asarray(manifest["grid"], dtype=float)
    truth = {
        name: np.exp(dgp.truth_curve(name, grid, centering).log_effect)
        for name in EXPOSURES
    }
    grids = {name: grid for name in EXPOSURES}
    result = StudyResult(truth=truth, grids=grids)

    for rec in manifest["datasets"]:
        estimates = {}
        missing = False
        for name in EXPOSURES:
            rel = _prefixed(rec["cumulative"][name], prefix)
            path = os.path.join(out_dir, rel)
            if not os.path.exists(path):
                missing = True
                break
            df = pd.read_csv(path)
            estimates[name] = {
                "mean": df["fit"].values,
                "lo": df["lo"].values,
                "hi": df["hi"].values,
            }
        if not missing:
            result.replicates.append(
                ReplicateResult(seed=int(rec.get("seed", rec["rep"])), estimates=estimates)
            )
    return result


def load_result_dict(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def require_replicates(study, label, exposure):
    n = len(study.replicates)
    if n != N_REPS:
        raise SystemExit(
            f"{label} produced {n}/{N_REPS} replicate curve(s) for {exposure}. "
            "Check the R output and tdlnm_fit_status.json before scoring."
        )


def r_timing_summary(r_timing, method, *, exposure=None):
    records = (r_timing or {}).get("records", [])
    keep = []
    by_rep = {}
    for rec in records:
        if rec.get("method") != method:
            continue
        if exposure is not None and rec.get("exposure") != exposure:
            continue
        if "fit_seconds" in rec:
            val = float(rec["fit_seconds"])
            keep.append(val)
            if exposure is None:
                by_rep[int(rec.get("rep", len(by_rep)))] = (
                    by_rep.get(int(rec.get("rep", len(by_rep))), 0.0) + val
                )
    if exposure is None and by_rep:
        keep = list(by_rep.values())
    if not keep:
        return {}
    return {
        "fit_seconds_mean": float(np.mean(keep)),
        "fit_seconds_sd": float(np.std(keep, ddof=1)) if len(keep) > 1 else 0.0,
        "fit_seconds_total": float(np.sum(keep)),
        "n_replicates": int(len(keep)),
        "source": "R timing.json",
        "aggregation": "sum_by_replicate" if exposure is None else "target_exposure_fit",
    }


def print_timing_tables(timing, exposures, models):
    print(f"\n=== joint runtime, seconds per replicate (R={N_REPS}) ===")
    print(f"{'fit':18s} " + " ".join(f"{m:>12s}" for m in models))
    cells = []
    for m in models:
        val = timing.get("joint", {}).get(m, {}).get("fit_seconds_mean")
        cells.append("-" if val is None else f"{float(val):.1f}")
    print(f"{'joint':18s} " + " ".join(f"{c:>12s}" for c in cells))

    by_exposure = timing.get("by_exposure", {})
    present = [
        m for m in models
        if any(m in by_exposure.get(name, {}) for name in exposures)
    ]
    if not present:
        return
    print(f"\n=== target-exposure runtime, seconds per fit (R={N_REPS}) ===")
    print(f"{'exposure':18s} " + " ".join(f"{m:>12s}" for m in present))
    for name in exposures:
        cells = []
        for m in present:
            val = by_exposure.get(name, {}).get(m, {}).get("fit_seconds_mean")
            cells.append("-" if val is None else f"{float(val):.1f}")
        print(f"{bp.NAMES.get(name, name):18s} " + " ".join(f"{c:>12s}" for c in cells))


def degradation_against_baseline(joint_results, baseline_results, exposures):
    out = {}
    for name in exposures:
        if name not in joint_results or name not in baseline_results:
            continue
        out[name] = {}
        for model, vals in joint_results[name].items():
            if model not in baseline_results[name]:
                continue
            out[name][model] = {}
            for tag in ("tot", "int", "bnd"):
                j = float(vals[f"err_{tag}"])
                b = float(baseline_results[name][model][f"err_{tag}"])
                js = float(vals.get(f"err_{tag}_se", 0.0))
                bs = float(baseline_results[name][model].get(f"err_{tag}_se", 0.0))
                if b <= 0 or not np.isfinite(b):
                    out[name][model][f"deg_{tag}"] = np.nan
                    out[name][model][f"deg_{tag}_se"] = np.nan
                    continue
                ratio = j / b
                rel_j = (js / j) if j > 0 else 0.0
                rel_b = bs / b
                out[name][model][f"deg_{tag}"] = ratio
                out[name][model][f"deg_{tag}_se"] = ratio * np.sqrt(rel_j ** 2 + rel_b ** 2)
    return out or None


def leakage_from_null(results):
    if "null" not in results:
        return None
    leakage = {}
    for model, vals in results["null"].items():
        leakage[model] = {
            tag: {
                "rmse": vals[f"err_{tag}"],
                "rmse_se": vals.get(f"err_{tag}_se", 0.0),
                "coverage": vals[f"cov_{tag}"],
                "coverage_se": vals.get(f"cov_{tag}_se", 0.0),
            }
            for tag in ("tot", "int", "bnd")
        }
    return leakage


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = results_dir(here)

    dgp = MX.joint_dgp(lag_max=LAG, include_null=INCLUDE_NULL)
    grid = np.linspace(*MX.VALUE_RANGE, 200)
    cen = Centering(method="reference", value=REF)

    export_joint_datasets(
        dgp,
        BENCH_DIR,
        n_reps=N_REPS,
        n_obs=N_OBS,
        grid=grid,
        reference=REF,
        base_seed=SEED,
    )

    rscript = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dlnam_bench",
        "fit_joint_mc.R",
    )
    print(
        f"Running joint DLNM-family fits in R ({', '.join(R_MODELS)}):\n"
        f"    {RSCRIPT} {rscript} {BENCH_DIR} {R_METHODS}\n"
    )
    r_start = time.perf_counter()
    r = subprocess.run([RSCRIPT, rscript, BENCH_DIR, R_METHODS],
                       capture_output=True, text=True)
    r_wall_seconds = time.perf_counter() - r_start
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"R exited with status {r.returncode} (see message above)")
    r_timing = load_json_if_exists(os.path.join(BENCH_DIR, "timing.json"))

    tcfg = TrainConfig(
        epochs=EPOCHS,
        n_ensemble=N_ENSEMBLE,
        lr=8e-4,
        lr_min=1e-4,
        weight_decay=1e-4,
        schedule="cosine",
        grad_clip=10,
    )
    dlnam_study = MonteCarloStudy(
        dgp=dgp,
        model_config=dlnam_config(LAG),
        train_config=tcfg,
        centering=cen,
        n_reps=N_REPS,
        n_obs=N_OBS,
        base_seed=SEED,
        device=DEVICE,
        se_source="laplace",
    ).run(progress=True)

    dlnm_studies = {
        label: load_joint_dlnm_study(BENCH_DIR, dgp, cen, prefix=spec["prefix"])
        for label, spec in R_MODELS.items()
    }

    base_sim = dgp.simulate(N_OBS, SEED)
    results, curves, boundary = {}, {}, {}
    for name in EXPOSURES:
        xvals = base_sim.frame[name].values
        q_lo, q_hi = np.quantile(xvals, [0.05, 0.95])
        boundary[name] = (float(q_lo), float(q_hi))
        m_bnd = (grid < q_lo) | (grid > q_hi)
        m_int = ~m_bnd

        row = {"DLNAM": summarise_term(dlnam_study, name, m_int, m_bnd)}
        for label, st in dlnm_studies.items():
            require_replicates(st, label, name)
            row[label] = summarise_term(st, name, m_int, m_bnd)
        results[name] = row

        cv = {
            "grid": np.asarray(dlnam_study.grids[name]),
            "truth": np.asarray(dlnam_study.truth[name]),
            "DLNAM": np.asarray(dlnam_study._stack(name, "mean").mean(0)),
        }
        for label, st in dlnm_studies.items():
            if st.replicates:
                cv[label] = np.asarray(st._stack(name, "mean").mean(0))
        curves[name] = cv

        d = row["DLNAM"]
        print(
            f"[{bp.NAMES.get(name, name):18s}] DLNAM err "
            f"{d['err_tot']:.4f} ± {d['err_tot_se']:.4f} "
            f"bias^2 {d['bias2_tot']:.2e} var {d['var_tot']:.2e} "
            f"cov {d['cov_tot']:.2f}"
        )
        for label in R_MODELS:
            m = row[label]
            print(
                f"                    {label:9s} err "
                f"{m['err_tot']:.4f} ± {m['err_tot_se']:.4f} "
                f"bias^2 {m['bias2_tot']:.2e} var {m['var_tot']:.2e} "
                f"cov {m['cov_tot']:.2f}"
            )

    baseline_path = out_dir / "mc_model_comparison.json"
    baseline_results = load_result_dict(baseline_path)
    degradation = degradation_against_baseline(
        results, baseline_results, PLOT_EXPOSURES
    ) if baseline_results is not None else None
    leakage = leakage_from_null(results)
    joint_timing = {"DLNAM": dlnam_study.timing_summary()}
    for label, spec in R_MODELS.items():
        ts = r_timing_summary(r_timing, spec["method"])
        if ts:
            joint_timing[label] = ts
    by_exposure_timing = {}
    for name in EXPOSURES:
        by_exposure_timing[name] = {}
        for label, spec in R_MODELS.items():
            ts = r_timing_summary(r_timing, spec["method"], exposure=name)
            if ts:
                by_exposure_timing[name][label] = ts
    timing = {
        "joint": joint_timing,
        "by_exposure": by_exposure_timing,
    }
    print_timing_tables(timing, PLOT_EXPOSURES, ["DLNAM", *R_MODELS.keys()])

    if degradation is None:
        print(f"[degradation] skipped: run run_mc.py first to create {baseline_path}")
    else:
        print("[degradation] joint/single RMSE ratios computed from mc_model_comparison.json")
        for name in PLOT_EXPOSURES:
            if name not in degradation:
                continue
            cells = [
                f"{model}={degradation[name][model]['deg_tot']:.2f}x"
                for model in ["DLNAM", *R_MODELS.keys()]
                if model in degradation[name]
            ]
            print(f"  {bp.NAMES.get(name, name):18s} " + "  ".join(cells))

    if leakage is not None:
        print("[leakage] null-exposure RMSE and pointwise coverage")
        for model in ["DLNAM", *R_MODELS.keys()]:
            if model not in leakage:
                continue
            v = leakage[model]["tot"]
            print(
                f"  {bp.LABELS.get(model, model):14s} "
                f"RMSE={v['rmse']:.4f} ± {v['rmse_se']:.4f}  "
                f"coverage={v['coverage']:.2f} ± {v['coverage_se']:.2f}"
            )

    result_path = out_dir / "mc_joint.json"
    save_result_bundle(
        result_path,
        kind="joint_mc",
        settings={
            "n_reps": N_REPS, "n_obs": N_OBS, "epochs": EPOCHS,
            "n_ensemble": N_ENSEMBLE, "lag": LAG, "reference": REF,
            "seed": SEED, "value_range": list(MX.VALUE_RANGE),
            "effect_scale": MX.EFFECT_SCALE,
            "exposure_corr": MX.EXPOSURE_CORR,
            "value_df_grid": list(VALUE_DF_GRID),
            "lag_df_grid": list(LAG_DF_GRID),
            "penalized_value_df": PENALIZED_VALUE_DF,
            "penalized_lag_df": PENALIZED_LAG_DF,
            "r_methods": R_METHODS,
            "tdlnm_family": "gaussian_log1p",
            "tdlnm_burn": TDLNM_SETTINGS["burn"],
            "tdlnm_iter": TDLNM_SETTINGS["iter"],
            "tdlnm_thin": TDLNM_SETTINGS["thin"],
            "tdlnm_attempts": TDLNM_SETTINGS["attempts"],
            "tdlnm_exposure_splits": TDLNM_SETTINGS["exposure_splits"],
            "tdlnm_adjust_value_df": TDLNM_SETTINGS["adjust_value_df"],
            "tdlnm_adjust_lag_df": TDLNM_SETTINGS["adjust_lag_df"],
            "include_null": INCLUDE_NULL,
            "se_source": "laplace",
            "baseline_results": baseline_path if baseline_results is not None else None,
        },
        exposures=EXPOSURES,
        models=["DLNAM", *R_MODELS.keys()],
        results=results,
        boundary=boundary,
        curves=curves,
        timing=timing,
        r_timing=r_timing,
        r_subprocess_wall_seconds_total=r_wall_seconds,
        degradation=degradation,
        leakage=leakage,
        r_environment=load_json_if_exists(os.path.join(BENCH_DIR, "r_environment.json")),
        tdlnm_fit_status=load_json_if_exists(os.path.join(BENCH_DIR, "tdlnm_fit_status.json")),
    )
    print(f"saved {result_path}")

    plot_results = {name: results[name] for name in PLOT_EXPOSURES}
    plot_curves = {name: curves[name] for name in PLOT_EXPOSURES}
    plot_boundary = {name: boundary[name] for name in PLOT_EXPOSURES}
    if degradation is not None and hasattr(bp, "save_all_with_degradation"):
        paths = bp.save_all_with_degradation(
            plot_results,
            out_dir,
            degradation=degradation,
            scenarios=PLOT_EXPOSURES,
            curves=plot_curves,
            boundary=plot_boundary,
            stem="mc_joint",
        )
    else:
        paths = bp.save_all(
            plot_results,
            out_dir,
            scenarios=PLOT_EXPOSURES,
            curves=plot_curves,
            boundary=plot_boundary,
            stem="mc_joint",
        )
    for p in paths:
        print(f"saved {p}")


if __name__ == "__main__":
    main()

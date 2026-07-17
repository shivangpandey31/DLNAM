"""
run_mc.py -- Monte Carlo DLNAM vs DLNM/TDLNM comparison.

The runner uses paired simulated datasets for every estimator. Python exports
the data manifest, R fits QAIC/QBIC/penalised DLNMs and the TDLNM comparator,
and Python fits DLNAM before scoring every method with the same StudyResult
logic: RMSE, bias^2, variance, and coverage.

The DLNAM configuration is the reference architecture used across the paper
experiments. Set N_REPS small for a local smoke test; raise it for the final run.
"""
import sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from dlnam_bench import plots as _bp
from experiment_io import load_json_if_exists, results_dir, save_result_bundle

from dlnam import (ModelConfig, TrainConfig, LayerSpec, ActivationSpec, ExUSpec,
                   InitSpec)
from dlnam.config import SurfaceTermSpec
from dlnam.terms.base import Centering
from dlnam_sim.study import MonteCarloStudy
from dlnam_bench import export_datasets, load_dlnm_study
from dlnam_sim.scenarios import scenarios, VALUE_RANGE

# ----------------------------- SETTINGS -----------------------------------
SCENARIOS  = ["smooth", "delayed_peaks", "localized_peak",
              "tilting_threshold"]
LAG        = 14            # matches scenarios LAG_MAX
N_REPS     = 3             # <-- smoke test; raise to ~100 for the real run
N_OBS      = 5000
EPOCHS     = 2500
N_ENSEMBLE = 3
REF        = 20.0
SEED       = 0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BENCH_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_model_comparison_data")
RSCRIPT    = "Rscript"
VALUE_DF_GRID = tuple(range(2, 16))
LAG_DF_GRID   = tuple(range(2, 16))
PENALIZED_VALUE_DF = max(VALUE_DF_GRID)
PENALIZED_LAG_DF   = max(LAG_DF_GRID)
TDLNM_SETTINGS = {
    "burn": 5000,
    "iter": 15000,
    "thin": 5,
    "attempts": 10,
    "exposure_splits": 20,
}
R_MODELS = {
    "QAIC": {"method": "qaic", "prefix": "qaic_"},
    "QBIC": {"method": "qbic", "prefix": "qbic_"},
    "Penalised": {"method": "pen", "prefix": "pen_"},
    "TDLNM": {"method": "tdlnm", "prefix": "tree_"},
}
R_METHODS = ",".join(spec["method"] for spec in R_MODELS.values())
MODELS = ["DLNAM", *R_MODELS.keys()]
# --------------------------------------------------------------------------


def dlnam_config(lag):
    # Reference architecture used by the model-comparison experiments.
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl       = lambda: InitSpec(scheme="torch_linear")
    return ModelConfig(terms={"x": SurfaceTermSpec(
        layers=[LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl())],
        num_subnets=N_ENSEMBLE, scaling="minmax", lag_max=lag,
        input_exu=ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                          weight_std=0.5, surface_strategy="concat",
                          bias_init=exu_bias()),
        mix_init=mix_init())}, link="log")


def summarise(study, mask_int=None, mask_bnd=None):
    """For each metric (RMSE/bias^2/variance/coverage), return (value, SE) in each
    region (total/interior/boundary). Error is reported as RMSE on the logRR scale,
    consistent with squared bias and variance: per grid point MSE = bias^2 + variance, so
    the three columns form one coherent decomposition (RMSE = sqrt(mean MSE))."""
    regions = [("tot", None), ("int", mask_int), ("bnd", mask_bnd)]
    out = {}
    for tag, m in regions:
        if tag != "tot" and m is None:
            continue
        out[f"err_{tag}"], out[f"err_{tag}_se"] = study.rmse_mean_se("x", mask=m)
        out[f"bias2_{tag}"], out[f"bias2_{tag}_se"] = study.bias2_mean_se("x", mask=m)
        out[f"var_{tag}"], out[f"var_{tag}_se"] = study.variance_mean_se("x", mask=m)
        out[f"cov_{tag}"], out[f"cov_{tag}_se"] = study.coverage_mean_se("x", mask=m)
    return out


def _require_replicates(study, label, scenario):
    n = len(study.replicates)
    if n != N_REPS:
        raise SystemExit(
            f"{label} produced {n}/{N_REPS} replicate curve(s) for {scenario}. "
            f"Check the R output and tdlnm_fit_status.json before scoring."
        )


def _r_timing_summary(r_timing, method, *, scenario=None, exposure=None):
    records = (r_timing or {}).get("records", [])
    keep = []
    for rec in records:
        if rec.get("method") != method:
            continue
        if scenario is not None and rec.get("scenario") != scenario:
            continue
        if exposure is not None and rec.get("exposure") != exposure:
            continue
        if "fit_seconds" in rec:
            keep.append(float(rec["fit_seconds"]))
    if not keep:
        return {}
    return {
        "fit_seconds_mean": float(np.mean(keep)),
        "fit_seconds_sd": float(np.std(keep, ddof=1)) if len(keep) > 1 else 0.0,
        "fit_seconds_total": float(np.sum(keep)),
        "n_replicates": int(len(keep)),
        "source": "R timing.json",
    }


def _print_timing_table(timing, scenarios, models):
    print(f"\n=== runtime, seconds per replicate (R={N_REPS}) ===")
    print(f"{'scenario':18s} " + " ".join(f"{m:>12s}" for m in models))
    for s in scenarios:
        cells = []
        for m in models:
            val = timing.get(s, {}).get(m, {}).get("fit_seconds_mean")
            cells.append("-" if val is None else f"{float(val):.1f}")
        print(f"{s:18s} " + " ".join(f"{c:>12s}" for c in cells))


def main():
    scen = {s: scenarios(lag_max=LAG)[s] for s in SCENARIOS}
    grid = np.linspace(*VALUE_RANGE, 200)
    cen = Centering(method="reference", value=REF)

    # Phase 1: write datasets + manifest (paired seeds: base_seed + rep).
    export_datasets(scen, BENCH_DIR, n_reps=N_REPS, n_obs=N_OBS, grid=grid,
                    reference=REF, base_seed=SEED,
                    value_df_grid=VALUE_DF_GRID, lag_df_grid=LAG_DF_GRID,
                    penalized_value_df=PENALIZED_VALUE_DF,
                    penalized_lag_df=PENALIZED_LAG_DF,
                    tdlnm_burn=TDLNM_SETTINGS["burn"],
                    tdlnm_iter=TDLNM_SETTINGS["iter"],
                    tdlnm_thin=TDLNM_SETTINGS["thin"],
                    tdlnm_attempts=TDLNM_SETTINGS["attempts"],
                    tdlnm_exposure_splits=TDLNM_SETTINGS["exposure_splits"])

    # Phase 2: fit all R-side DLNM-family comparators on those datasets.
    rscript = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dlnam_bench", "fit_dlnm.R")
    print(f"Running DLNM-family fits in R ({', '.join(R_MODELS)}):\n    "
          f"{RSCRIPT} {rscript} {BENCH_DIR} {R_METHODS}\n")
    r_start = time.perf_counter()
    r = subprocess.run([RSCRIPT, rscript, BENCH_DIR, R_METHODS],
                       capture_output=True, text=True)
    r_wall_seconds = time.perf_counter() - r_start
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"R exited with status {r.returncode} (see message above)")
    r_timing = load_json_if_exists(os.path.join(BENCH_DIR, "timing.json"))

    # Phase 3: DLNAM Monte Carlo on the same data, then compare.
    tcfg = TrainConfig(epochs=EPOCHS, n_ensemble=N_ENSEMBLE, lr=8e-4, lr_min=1e-4,
                       weight_decay=1e-4, schedule="cosine", grad_clip=10)

    results = {}   # scenario -> {"DLNAM": summary, "QAIC": summary, ...}
    curves = {}    # scenario -> {"grid","truth","DLNAM",...} for the composite A-row
    boundary = {}  # scenario -> (q_lo, q_hi) exposure-boundary marks
    timing = {}
    for j, s in enumerate(SCENARIOS):
        # boundary mask from this scenario's sampled exposure at the base seed
        # (deterministic; same data the fits used). Interior = complement.
        xvals = scen[s].simulate(N_OBS, SEED).frame["x"].values
        q_lo, q_hi = np.quantile(xvals, 0.05), np.quantile(xvals, 0.95)
        boundary[s] = (float(q_lo), float(q_hi))
        bnd = (grid < q_lo) | (grid > q_hi)
        m_int, m_bnd = ~bnd, bnd

        dlnam_study = MonteCarloStudy(dgp=scen[s], model_config=dlnam_config(LAG),
                                      train_config=tcfg, centering=cen,
                                      n_reps=N_REPS, n_obs=N_OBS, base_seed=SEED,
                                      device=DEVICE, se_source="laplace").run(progress=False)
        timing[s] = {"DLNAM": dlnam_study.timing_summary()}
        for label, spec in R_MODELS.items():
            ts = _r_timing_summary(r_timing, spec["method"], scenario=s)
            if ts:
                timing[s][label] = ts
        row = {"DLNAM": summarise(dlnam_study, m_int, m_bnd)}
        dlnm_studies = {}
        for label, spec in R_MODELS.items():
            st = load_dlnm_study(BENCH_DIR, scen[s], s, cen, prefix=spec["prefix"])
            _require_replicates(st, label, s)
            dlnm_studies[label] = st
            row[label] = summarise(st, m_int, m_bnd)
        results[s] = row

        d = row["DLNAM"]
        print(f"[{s:9s}] DLNAM err {d['err_tot']:.4f} ± {d['err_tot_se']:.4f} "
              f"bias^2 {d['bias2_tot']:.2e} var {d['var_tot']:.2e} cov {d['cov_tot']:.2f}")
        for label in R_MODELS:
            m = row[label]
            print(f"            {label:9s} err {m['err_tot']:.4f} ± {m['err_tot_se']:.4f} "
                  f"bias^2 {m['bias2_tot']:.2e} var {m['var_tot']:.2e} cov {m['cov_tot']:.2f}")

        dmean = np.mean([r.estimates["x"]["mean"] for r in dlnam_study.replicates], 0)
        cv = {"grid": np.asarray(grid), "truth": np.asarray(dlnam_study.truth["x"]),
              "DLNAM": np.asarray(dmean)}
        for label in R_MODELS:
            st = dlnm_studies[label]
            if st.replicates:
                mmean = np.mean([r.estimates["x"]["mean"] for r in st.replicates], 0)
                cv[label] = np.asarray(mmean)
        curves[s] = cv

    # --- summary tables: every metric, every region, value ± analytical SE ----
    METRICS = [("error", "err", "{:.4f}"), ("bias^2", "bias2", "{:.2e}"),
               ("variance", "var", "{:.2e}"), ("coverage", "cov", "{:.2f}")]
    REGIONS = [("total", "tot"), ("interior", "int"), ("boundary", "bnd")]
    for mlabel, mkey, fmt in METRICS:
        print(f"\n=== {mlabel}, value ± SE (R={N_REPS}) ===")
        for rlabel, rtag in REGIONS:
            print(f"-- {rlabel} --")
            print(f"{'scenario':9s} " + " ".join(f"{k:>20s}" for k in MODELS))
            for s in SCENARIOS:
                cells = []
                for k in MODELS:
                    v = results[s][k][f"{mkey}_{rtag}"]
                    se = results[s][k][f"{mkey}_{rtag}_se"]
                    cells.append(f"{fmt.format(v)} ± {fmt.format(se)}")
                print(f"{s:9s} " + " ".join(f"{c:>20s}" for c in cells))

    _print_timing_table(timing, SCENARIOS, MODELS)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = results_dir(here)
    result_path = out_dir / "mc_model_comparison.json"
    save_result_bundle(
        result_path,
        kind="single_exposure_mc",
        settings={
            "n_reps": N_REPS, "n_obs": N_OBS, "epochs": EPOCHS,
            "n_ensemble": N_ENSEMBLE, "lag": LAG, "reference": REF,
            "seed": SEED, "value_range": list(VALUE_RANGE),
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
            "se_source": "laplace",
        },
        scenarios=SCENARIOS,
        models=MODELS,
        results=results,
        boundary=boundary,
        curves=curves,
        timing=timing,
        r_timing=r_timing,
        r_subprocess_wall_seconds_total=r_wall_seconds,
        r_environment=load_json_if_exists(os.path.join(BENCH_DIR, "r_environment.json")),
        tdlnm_fit_status=load_json_if_exists(os.path.join(BENCH_DIR, "tdlnm_fit_status.json")),
    )
    print(f"saved {result_path}")

    # publication metric figures (forest RMSE, bias^2/variance decomposition,
    # coverage) -- compact alternatives to the large numeric tables above.
    try:
        paths = _bp.save_all(results, out_dir, scenarios=list(SCENARIOS),
                             curves=curves, boundary=boundary,
                             stem="mc_model_comparison")
        for p in paths:
            print(f"saved {p}")
    except Exception as e:
        print(f"[plots] skipped figure export: {e}")


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    main()

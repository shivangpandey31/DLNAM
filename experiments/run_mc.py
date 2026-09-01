"""Monte Carlo comparison with selectable evaluation targets.

Examples
--------
Evaluate both targets:

    python experiments/run_mc.py --evaluation both

Evaluate only the complete exposure-lag surface:

    python experiments/run_mc.py --evaluation surface
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
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
from dlnam_bench import (
    export_datasets,
    load_dlnm_study,
    load_dlnm_surface_study,
)
from dlnam_bench import plots as bp
from dlnam_sim.scenarios import VALUE_RANGE, scenarios
from dlnam_sim.targets import (
    EVALUATION_CHOICES,
    evaluation_targets,
    run_target_studies,
    summarise_regions,
)
from experiment_io import load_json_if_exists, results_dir, save_result_bundle


SCENARIOS = ["dgp1", "dgp2", "dgp3", "dgp4"]
LAG = 14
N_REPS = 200
N_OBS = 5000
EPOCHS = 2500
N_ENSEMBLE = 3
N_SUBNETS = 3
REF = 20.0
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RSCRIPT = "Rscript"
EVALUATION = "both"
EXU_WEIGHT_MEAN = 1.5
EXU_LAG_WEIGHT_MEAN = 2.5
EXU_WEIGHT_STD = 0.5
LEARNING_RATE = 8e-4
MIN_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
# Cross-basis search grid for the criterion-selected DLNMs, and the fixed rank
# for the penalised fit. The grid starts at 2: one degree of freedom leaves no
# interior knot, so the marginal basis degenerates to a linear term.
VALUE_DF_GRID = tuple(range(2, 11))
LAG_DF_GRID = tuple(range(2, 11))
PENALIZED_VALUE_DF = 10
PENALIZED_LAG_DF = 10
# T-DLNM sampler settings, set explicitly so the fit does not depend on the
# defaults of the installed dlmtree version.
TDLNM_SETTINGS = {
    "burn": 5000,
    "iter": 15000,
    "thin": 10,
    "attempts": 10,
    "exposure_splits": 30,
    "trees": 20,
}
R_MODELS = {
    "QAIC": {"method": "qaic", "prefix": "qaic_"},
    "QBIC": {"method": "qbic", "prefix": "qbic_"},
    "Penalised": {"method": "pen", "prefix": "pen_"},
    "TDLNM": {"method": "tdlnm", "prefix": "tree_"},
}
R_METHODS = ",".join(spec["method"] for spec in R_MODELS.values())
MODELS = ["DLNAM", *R_MODELS]
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent / "mc_model_comparison_data"
)


def dlnam_config(
    lag: int,
    *,
    exu_weight_mean: float = EXU_WEIGHT_MEAN,
    exu_lag_weight_mean: float = EXU_LAG_WEIGHT_MEAN,
) -> ModelConfig:
    """Return the reference architecture used in the model comparison."""
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    torch_linear = lambda: InitSpec(scheme="torch_linear")
    return ModelConfig(
        terms={
            "x": SurfaceTermSpec(
                layers=[
                    LayerSpec(128, mish()),
                    LayerSpec(
                        128,
                        mish(),
                        weight_init=torch_linear(),
                        bias_init=torch_linear(),
                    ),
                ],
                num_subnets=N_SUBNETS,
                scaling="minmax",
                lag_max=lag,
                input_exu=ExUSpec(
                    enabled=True,
                    weight_mean=exu_weight_mean,
                    weight_mean_lag=exu_lag_weight_mean,
                    weight_std=EXU_WEIGHT_STD,
                    surface_strategy="concat",
                    bias_init=exu_bias(),
                ),
                mix_init=mix_init(),
            )
        },
        link="log",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the DLNAM versus DLNM-family comparison with cumulative, "
            "surface, or both evaluation targets."
        )
    )
    parser.add_argument(
        "--evaluation",
        choices=EVALUATION_CHOICES,
        default=EVALUATION,
    )
    parser.add_argument("--n-reps", type=int, default=N_REPS)
    parser.add_argument("--n-obs", type=int, default=N_OBS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rscript", default=RSCRIPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    for name in ("n_reps", "n_obs", "epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def _require_replicates(study, expected, label, scenario, target):
    actual = len(study.replicates)
    if actual != expected:
        raise SystemExit(
            f"{label} produced {actual}/{expected} {target} replicate(s) for "
            f"{scenario}. Run without --skip-r and inspect the R status files."
        )


def _r_timing_summary(r_timing, method, *, scenario=None):
    records = (r_timing or {}).get("records", [])
    values = []
    for record in records:
        if record.get("method") != method:
            continue
        if scenario is not None and record.get("scenario") != scenario:
            continue
        if "fit_seconds" in record:
            values.append(float(record["fit_seconds"]))
    if not values:
        return {}
    return {
        "fit_seconds_mean": float(np.mean(values)),
        "fit_seconds_sd": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        "fit_seconds_total": float(np.sum(values)),
        "n_replicates": len(values),
        "source": "R timing.json",
    }


def _surface_matrix(values, grid):
    """Reshape a lag-major flattened surface into lag x exposure form."""
    arr = np.asarray(values, dtype=float)
    n_grid = len(grid)
    if arr.size % n_grid != 0:
        raise ValueError(
            f"surface vector of length {arr.size} is incompatible with "
            f"grid length {n_grid}"
        )
    return arr.reshape(arr.size // n_grid, n_grid)


def _print_results(results, target, n_reps):
    print(f"\n=== {target}, value +/- MCSE (R={n_reps}) ===")
    for scenario in SCENARIOS:
        print(f"\n[{scenario}]")
        for model in MODELS:
            row = results[scenario][model]
            print(
                f"  {model:10s} "
                f"RMSE {row['err_tot']:.5f} +/- {row['err_tot_se']:.5f}  "
                f"bias^2 {row['bias2_tot']:.2e}  "
                f"variance {row['var_tot']:.2e}  "
                f"coverage {row['cov_tot']:.3f}"
            )


def _print_timing_table(timing, n_reps):
    print(f"\n=== runtime, seconds per replicate (R={n_reps}) ===")
    print(f"{'scenario':18s} " + " ".join(f"{m:>12s}" for m in MODELS))
    for scenario in SCENARIOS:
        cells = []
        for model in MODELS:
            value = timing.get(scenario, {}).get(model, {}).get(
                "fit_seconds_mean"
            )
            cells.append("-" if value is None else f"{float(value):.1f}")
        print(
            f"{scenario:18s} "
            + " ".join(f"{cell:>12s}" for cell in cells)
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    here = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    scenario_specs = {
        name: scenarios(lag_max=LAG)[name] for name in SCENARIOS
    }
    # 201 points, not 200, so the step is exactly 0.2 and REF=20.0 falls on a
    # grid node. Centering.anchor snaps the reference to the nearest grid point
    # when centring the cumulative curve, whereas the surface truth and the R
    # estimators centre at the exact reference; off-grid the two conventions
    # differ by f(nearest)-f(REF), a constant offset applied to the comparators
    # but not to the DLNAM. On-grid the conventions coincide and the offset is
    # identically zero.
    grid = np.linspace(*VALUE_RANGE, 201)
    assert np.isclose(grid, REF).any(), "reference must lie on the scoring grid"
    centering = Centering(method="reference", value=REF)
    targets = evaluation_targets(args.evaluation)

    export_datasets(
        scenario_specs,
        str(data_dir),
        n_reps=args.n_reps,
        n_obs=args.n_obs,
        grid=grid,
        reference=REF,
        base_seed=args.seed,
        value_df_grid=VALUE_DF_GRID,
        lag_df_grid=LAG_DF_GRID,
        penalized_value_df=PENALIZED_VALUE_DF,
        penalized_lag_df=PENALIZED_LAG_DF,
        tdlnm_burn=TDLNM_SETTINGS["burn"],
        tdlnm_iter=TDLNM_SETTINGS["iter"],
        tdlnm_thin=TDLNM_SETTINGS["thin"],
        tdlnm_attempts=TDLNM_SETTINGS["attempts"],
        tdlnm_exposure_splits=TDLNM_SETTINGS["exposure_splits"],
        tdlnm_trees=TDLNM_SETTINGS["trees"],
    )

    r_script = here.parent / "dlnam_bench" / "fit_dlnm.R"
    command = [args.rscript, str(r_script), str(data_dir), R_METHODS]
    print(
        "Running DLNM-family fits in R "
        f"({', '.join(R_MODELS)}):\n    {' '.join(command)}\n"
    )
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    r_wall_seconds = time.perf_counter() - started
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise SystemExit(
            f"R exited with status {completed.returncode} "
            "(see message above)"
        )

    r_timing = load_json_if_exists(data_dir / "timing.json")
    train_config = TrainConfig(
        epochs=args.epochs,
        n_ensemble=N_ENSEMBLE,
        lr=LEARNING_RATE,
        lr_min=MIN_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        schedule="cosine",
        grad_clip=GRAD_CLIP,
    )
    target_results = {target: {} for target in targets}
    curves = {}
    surface_curves = {}
    boundary = {}
    timing = {}

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario}")
        dgp = scenario_specs[scenario]
        exposure = dgp.simulate(args.n_obs, args.seed).frame["x"].to_numpy()
        q_lo, q_hi = np.quantile(exposure, [0.05, 0.95])
        boundary[scenario] = (float(q_lo), float(q_hi))
        cumulative_boundary = (grid < q_lo) | (grid > q_hi)
        surface_boundary = np.tile(cumulative_boundary, LAG + 1)

        dlnam_studies = run_target_studies(
            dgp=dgp,
            model_config=dlnam_config(LAG),
            train_config=train_config,
            centering=centering,
            evaluation=args.evaluation,
            n_reps=args.n_reps,
            n_obs=args.n_obs,
            base_seed=args.seed,
            se_source="laplace+ensemble",
            device=args.device,
            progress=True,
        )
        timing[scenario] = {
            "DLNAM": dlnam_studies[targets[0]].timing_summary()
        }
        for label, spec in R_MODELS.items():
            summary = _r_timing_summary(
                r_timing,
                spec["method"],
                scenario=scenario,
            )
            if summary:
                timing[scenario][label] = summary

        rows = {target: {} for target in targets}
        for target, study in dlnam_studies.items():
            _require_replicates(
                study,
                args.n_reps,
                "DLNAM",
                scenario,
                target,
            )
            mask = (
                surface_boundary
                if target == "surface"
                else cumulative_boundary
            )
            rows[target]["DLNAM"] = summarise_regions(
                study,
                interior=~mask,
                boundary=mask,
            )

        curve_payload = None
        if "cumulative" in dlnam_studies:
            study = dlnam_studies["cumulative"]
            curve_payload = {
                "grid": np.asarray(study.grids["x"]),
                "truth": np.asarray(study.truth["x"]),
                "DLNAM": study._stack("x", "mean").mean(0),
            }
        surface_payload = None
        if "surface" in dlnam_studies:
            study = dlnam_studies["surface"]
            surface_payload = {
                "grid": np.asarray(grid),
                "lags": np.arange(LAG + 1, dtype=float),
                "truth": _surface_matrix(study.truth["x"], grid),
                "DLNAM": _surface_matrix(
                    study._stack("x", "mean").mean(0),
                    grid,
                ),
            }

        for label, spec in R_MODELS.items():
            for target in targets:
                if target == "surface":
                    study = load_dlnm_surface_study(
                        str(data_dir),
                        dgp,
                        scenario,
                        centering,
                        prefix=spec["prefix"],
                    )
                    mask = surface_boundary
                else:
                    study = load_dlnm_study(
                        str(data_dir),
                        dgp,
                        scenario,
                        centering,
                        prefix=spec["prefix"],
                    )
                    mask = cumulative_boundary
                _require_replicates(
                    study,
                    args.n_reps,
                    label,
                    scenario,
                    target,
                )
                rows[target][label] = summarise_regions(
                    study,
                    interior=~mask,
                    boundary=mask,
                )
                if target == "cumulative" and curve_payload is not None:
                    curve_payload[label] = study._stack("x", "mean").mean(0)
                if target == "surface" and surface_payload is not None:
                    surface_payload[label] = _surface_matrix(
                        study._stack("x", "mean").mean(0),
                        grid,
                    )

        for target in targets:
            target_results[target][scenario] = rows[target]
        if curve_payload is not None:
            curves[scenario] = curve_payload
        if surface_payload is not None:
            surface_curves[scenario] = surface_payload

    for target in targets:
        _print_results(target_results[target], target, args.n_reps)
    _print_timing_table(timing, args.n_reps)

    primary = (
        target_results["cumulative"]
        if "cumulative" in target_results
        else target_results["surface"]
    )
    extra_payload = {}
    if "surface" in target_results:
        extra_payload["surface_results"] = target_results["surface"]
    if surface_curves:
        extra_payload["surface_curves"] = surface_curves

    output_dir = results_dir(here)
    output = output_dir / "mc_model_comparison.json"
    save_result_bundle(
        output,
        kind="single_exposure_mc",
        settings={
            "evaluation": args.evaluation,
            "n_reps": args.n_reps,
            "n_obs": args.n_obs,
            "epochs": args.epochs,
            "n_ensemble": N_ENSEMBLE,
            "n_subnets": N_SUBNETS,
            "hidden_widths": [128, 128],
            "activation": "Mish",
            "exu_strategy": "concat",
            "exu_weight_mean": EXU_WEIGHT_MEAN,
            "exu_lag_weight_mean": EXU_LAG_WEIGHT_MEAN,
            "exu_weight_std": EXU_WEIGHT_STD,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "schedule": "cosine",
            "gradient_clip": GRAD_CLIP,
            "lag": LAG,
            "reference": REF,
            "seed": args.seed,
            "device": args.device,
            "value_range": list(VALUE_RANGE),
            "n_value_grid": len(grid),
            "n_surface_points": len(grid) * (LAG + 1),
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
            "tdlnm_trees": TDLNM_SETTINGS["trees"],
            "se_source": "laplace+ensemble",
        },
        scenarios=SCENARIOS,
        models=MODELS,
        results=primary,
        boundary=boundary,
        curves=curves,
        timing=timing,
        r_timing=r_timing,
        r_subprocess_wall_seconds_total=r_wall_seconds,
        r_environment=load_json_if_exists(data_dir / "r_environment.json"),
        tdlnm_fit_status=load_json_if_exists(
            data_dir / "tdlnm_fit_status.json"
        ),
        **extra_payload,
    )
    print(f"\nsaved {output}")

    if "cumulative" in target_results:
        try:
            paths = bp.save_all(
                target_results["cumulative"],
                output_dir,
                scenarios=list(SCENARIOS),
                curves=curves,
                boundary=boundary,
                stem="mc_model_comparison_cumulative",
                title="Simulation Study: Model Comparison",
            )
            for path in paths:
                print(f"saved {path}")
        except Exception as error:
            print(f"[plots] skipped cumulative figure export: {error}")

    if "surface" in target_results:
        try:
            paths = bp.save_metric_composite(
                target_results["surface"],
                output_dir,
                scenarios=list(SCENARIOS),
                stem="mc_model_comparison_surface",
                title="Simulation Study: Model Comparison (Surface)",
            )
            for path in paths:
                print(f"saved {path}")
        except Exception as error:
            print(f"[plots] skipped surface figure export: {error}")


if __name__ == "__main__":
    main()

"""DLNAM architecture ablation with selectable evaluation targets."""
from __future__ import annotations

import argparse
import os
import sys
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
from dlnam_bench import plots as bp
from dlnam_sim.scenarios import LAG_MAX, VALUE_RANGE, scenarios
from dlnam_sim.targets import (
    EVALUATION_CHOICES,
    evaluation_targets,
    run_target_studies,
    summarise_regions,
)
from experiment_io import results_dir, save_result_bundle


SCENARIOS = ["dgp1", "dgp2", "dgp3", "dgp4"]
N_REPS, N_OBS, EPOCHS, SEED = 50, 5000, 2500, 0
N_ENSEMBLE = 3
N_SUBNETS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF = 20.0
EVALUATION = "both"
EXU_WEIGHT_MEAN = 1.5
EXU_LAG_WEIGHT_MEAN = 2.5
EXU_WEIGHT_STD = 0.5
LEARNING_RATE = 8e-4
MIN_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0

_SEQUENCE = [
    bp.COLOURS[model]
    for model in ("DLNAM", "QAIC", "QBIC", "Penalised")
]
_MARKERS = ["o", "^", "s", "D"]
ABLATIONS = ["reference", "no_exu", "no_subnets", "no_smooth"]
ABL_COLOURS = {
    key: _SEQUENCE[index] for index, key in enumerate(ABLATIONS)
}
ABL_MARKERS = {
    key: _MARKERS[index] for index, key in enumerate(ABLATIONS)
}
ABL_LABELS = {
    "reference": "Reference",
    "no_exu": "ExU Ablation",
    "no_subnets": "Mixture Ablation",
    "no_smooth": "Smoothness Ablation",
}


def base_surface_spec(
    lag,
    *,
    exu=True,
    subnets=N_SUBNETS,
    activation=None,
    mix_init="normal",
):
    """Return the reference surface architecture with selected components."""
    activation_factory = (
        activation
        if activation is not None
        else lambda: ActivationSpec(base=torch.nn.Mish)
    )
    mixing = (
        None
        if mix_init is None
        else InitSpec(scheme="normal", mean=0.0, std=0.1)
    )
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    torch_linear = lambda: InitSpec(scheme="torch_linear")
    input_exu = (
        ExUSpec(
            enabled=True,
            weight_mean=EXU_WEIGHT_MEAN,
            weight_mean_lag=EXU_LAG_WEIGHT_MEAN,
            weight_std=EXU_WEIGHT_STD,
            surface_strategy="concat",
            bias_init=exu_bias(),
        )
        if exu
        else ExUSpec(enabled=False)
    )
    return SurfaceTermSpec(
        layers=[
            LayerSpec(128, activation_factory()),
            LayerSpec(
                128,
                activation_factory(),
                weight_init=torch_linear(),
                bias_init=torch_linear(),
            ),
        ],
        num_subnets=subnets,
        scaling="minmax",
        lag_max=lag,
        input_exu=input_exu,
        mix_init=mixing,
    )


def config_for(key, lag):
    """Return the model configuration for one ceteris-paribus ablation."""
    relu1 = lambda: ActivationSpec(
        base=lambda: torch.nn.Hardtanh(0.0, 1.0)
    )
    if key == "reference":
        spec = base_surface_spec(lag)
    elif key == "no_exu":
        spec = base_surface_spec(lag, exu=False)
    elif key == "no_subnets":
        spec = base_surface_spec(lag, subnets=1, mix_init=None)
    elif key == "no_smooth":
        spec = base_surface_spec(lag, activation=relu1)
    else:
        raise ValueError(key)
    return ModelConfig(terms={"x": spec}, link="log")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the DLNAM architecture ablation with cumulative, surface, "
            "or both evaluation targets."
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
    args = parser.parse_args()
    for name in ("n_reps", "n_obs", "epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def _print_row(scenario, label, target, row):
    print(
        f"[{scenario:18s}] {label:20s} {target:10s} "
        f"RMSE {row['err_tot']:.4f} +/- {row['err_tot_se']:.4f}  "
        f"bias^2 {row['bias2_tot']:.2e}  "
        f"variance {row['var_tot']:.2e}  "
        f"coverage {row['cov_tot']:.3f}"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    here = Path(__file__).resolve().parent
    scenario_specs = scenarios(lag_max=LAG_MAX)
    grid = np.linspace(*VALUE_RANGE, 201)
    centering = Centering(method="reference", value=REF)
    targets = evaluation_targets(args.evaluation)
    target_results = {target: {} for target in targets}
    curves = {}
    boundary = {}
    timing = {}

    for scenario in SCENARIOS:
        dgp = scenario_specs[scenario]
        exposure = dgp.simulate(args.n_obs, args.seed).frame["x"].to_numpy()
        q_lo, q_hi = np.quantile(exposure, [0.05, 0.95])
        boundary[scenario] = (float(q_lo), float(q_hi))
        curve_boundary = (grid < q_lo) | (grid > q_hi)
        surface_boundary = np.tile(curve_boundary, LAG_MAX + 1)

        rows = {target: {} for target in targets}
        curve_payload = None
        scenario_timing = {}
        for key in ABLATIONS:
            train_config = TrainConfig(
                epochs=args.epochs,
                n_ensemble=N_ENSEMBLE,
                lr=LEARNING_RATE,
                lr_min=MIN_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                schedule="cosine",
                grad_clip=GRAD_CLIP,
                seed=args.seed,
            )
            studies = run_target_studies(
                dgp=dgp,
                model_config=config_for(key, LAG_MAX),
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
            for target, study in studies.items():
                mask = (
                    surface_boundary
                    if target == "surface"
                    else curve_boundary
                )
                rows[target][key] = summarise_regions(
                    study,
                    interior=~mask,
                    boundary=mask,
                )
                _print_row(
                    scenario,
                    ABL_LABELS[key],
                    target,
                    rows[target][key],
                )

            timing_study = studies[targets[0]]
            scenario_timing[key] = timing_study.timing_summary()
            if "cumulative" in studies:
                study = studies["cumulative"]
                if curve_payload is None:
                    curve_payload = {
                        "grid": np.asarray(study.grids["x"]),
                        "truth": np.asarray(study.truth["x"]),
                    }
                curve_payload[key] = study._stack("x", "mean").mean(0)

        for target in targets:
            target_results[target][scenario] = rows[target]
        if curve_payload is not None:
            curves[scenario] = curve_payload
        timing[scenario] = scenario_timing

    primary = (
        target_results["cumulative"]
        if "cumulative" in target_results
        else target_results["surface"]
    )
    payload = {}
    if "surface" in target_results:
        payload["surface_results"] = target_results["surface"]

    output_dir = results_dir(here)
    output = output_dir / "mc_ablation.json"
    save_result_bundle(
        output,
        kind="dlnam_architecture_ablation_mc",
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
            "lag": LAG_MAX,
            "reference": REF,
            "seed": args.seed,
            "device": args.device,
            "value_range": list(VALUE_RANGE),
            "n_value_grid": len(grid),
            "n_surface_points": len(grid) * (LAG_MAX + 1),
            "se_source": "laplace+ensemble",
            "labels": ABL_LABELS,
        },
        scenarios=SCENARIOS,
        models=list(ABLATIONS),
        results=primary,
        boundary=boundary,
        curves=curves,
        timing=timing,
        **payload,
    )
    print(f"saved {output}")

    old_colours, old_markers = bp.COLOURS, bp.MARKERS
    old_labels, old_models = bp.LABELS, bp.MODELS
    bp.COLOURS, bp.MARKERS = ABL_COLOURS, ABL_MARKERS
    bp.LABELS, bp.MODELS = ABL_LABELS, list(ABLATIONS)
    try:
        if "cumulative" in target_results:
            paths = bp.save_all(
                target_results["cumulative"],
                output_dir,
                scenarios=SCENARIOS,
                curves=curves,
                boundary=boundary,
                stem="mc_ablation_cumulative",
                title="Simulation Study: Architecture Ablation",
            )
            for path in paths:
                print(f"saved {path}")
        if "surface" in target_results:
            # Same renderer as the model-comparison surface figure, so all three
            # surface figures share one row-label layout.
            paths = bp.save_metric_composite(
                target_results["surface"],
                output_dir,
                scenarios=SCENARIOS,
                stem="mc_ablation_surface",
                title="Simulation Study: Architecture Ablation (Surface)",
            )
            for path in paths:
                print(f"saved {path}")
    finally:
        bp.COLOURS, bp.MARKERS = old_colours, old_markers
        bp.LABELS, bp.MODELS = old_labels, old_models


if __name__ == "__main__":
    main()

"""Monte Carlo evaluation of cumulative effects and complete lag surfaces."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from dlnam import (DataProcessor, EffectExtractor, IntervalUQ, TrainConfig,
                   Trainer, needs_laplace)
from dlnam.config import ModelConfig
from dlnam.links import make_link
from dlnam.terms.base import Centering

from .dgp import DataGeneratingProcess
from .study import ReplicateResult, StudyResult


EVALUATION_CHOICES = ("cumulative", "surface", "both")


def evaluation_targets(evaluation: str) -> tuple[str, ...]:
    """Resolve a command-line evaluation mode into concrete targets."""
    if evaluation not in EVALUATION_CHOICES:
        raise ValueError(
            f"evaluation must be one of {EVALUATION_CHOICES}, got {evaluation!r}"
        )
    return (
        ("cumulative", "surface") if evaluation == "both" else (evaluation,)
    )


def _surface_reference(term, centering: Centering) -> float:
    if centering.method in ("reference", "custom"):
        if centering.value is None:
            raise ValueError(f"centering={centering.method!r} requires a value")
        return float(centering.value)
    if centering.method == "median":
        return float(term._data_median)
    raise ValueError(
        "surface evaluation supports reference, custom, or median centering"
    )


def _surface_truth(dgp, name, grid, centering, link):
    term = dgp.true_terms[name]
    if term.kind != "surface":
        raise ValueError(f"{name!r} is not a surface term")
    reference = _surface_reference(term, centering)
    lags = np.arange(term.lag_max + 1, dtype=float)
    raw = np.asarray(term.fn(grid[:, None], lags[None, :]), dtype=float)
    ref = np.asarray(
        term.fn(np.asarray([[reference]], dtype=float), lags[None, :]),
        dtype=float,
    )
    log_effect = (raw - ref).T
    effect = (
        np.exp(log_effect)
        if link.name in ("log", "logit")
        else log_effect
    )
    return effect.reshape(-1), np.tile(grid, len(lags))


def _initial_results(
    dgp: DataGeneratingProcess,
    model_config: ModelConfig,
    centering: Centering,
    targets: Iterable[str],
) -> dict[str, StudyResult]:
    link = make_link(model_config.link)
    results = {}
    for target in targets:
        grids = {}
        truth = {}
        for name in model_config.terms:
            if name not in dgp.true_terms:
                continue
            term = dgp.true_terms[name]
            grid = term.default_grid()
            if target == "surface":
                if term.kind != "surface":
                    continue
                truth[name], grids[name] = _surface_truth(
                    dgp, name, grid, centering, link
                )
            else:
                curve = dgp.truth_curve(name, grid, centering)
                grids[name] = grid
                truth[name] = (
                    np.exp(curve.log_effect)
                    if link.name in ("log", "logit")
                    else curve.log_effect
                )
        results[target] = StudyResult(truth=truth, grids=grids)
    return results


def run_target_studies(
    *,
    dgp: DataGeneratingProcess,
    model_config: ModelConfig,
    train_config: TrainConfig,
    centering: Centering,
    evaluation: str = "both",
    n_reps: int = 500,
    n_obs: int = 5000,
    alpha: float = 0.05,
    base_seed: int = 0,
    se_source: str = "laplace",
    device: str = "cpu",
    progress: bool = True,
) -> dict[str, StudyResult]:
    """Fit each replicate once and evaluate the requested inferential targets."""
    targets = evaluation_targets(evaluation)
    results = _initial_results(dgp, model_config, centering, targets)
    link = make_link(model_config.link)

    iterator = range(n_reps)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_reps)
        except ImportError:
            pass

    for rep in iterator:
        seed = base_seed + rep
        simulated = dgp.simulate(n_obs, seed)
        replicate_config = TrainConfig(
            **{**train_config.__dict__, "seed": seed}
        )
        trainer = Trainer(
            model_config,
            replicate_config,
            device=torch.device(device),
        )
        processor = DataProcessor(model_config)
        prepared = processor.prepare(
            simulated.frame,
            trainer.ensemble,
            dgp.target_col,
        )
        trainer.fit(prepared.inputs, prepared.y)

        if needs_laplace(se_source):
            extractor = EffectExtractor.with_laplace(
                trainer.ensemble,
                prepared,
                link,
                centering,
                interval=se_source,
            )
        else:
            extractor = EffectExtractor(
                trainer.ensemble,
                link,
                IntervalUQ(se_source),
                centering,
            )

        for target in targets:
            estimates = {}
            components = {}
            for name, stored_grid in results[target].grids.items():
                true_term = dgp.true_terms[name]
                grid = (
                    true_term.default_grid()
                    if target == "surface"
                    else stored_grid
                )
                estimate = (
                    extractor.extract_surface(name, grid, alpha=alpha)
                    if target == "surface"
                    else extractor.extract(name, grid, alpha=alpha)
                )
                estimates[name] = {
                    "mean": np.asarray(estimate.mean).reshape(-1),
                    "lo": np.asarray(estimate.lo).reshape(-1),
                    "hi": np.asarray(estimate.hi).reshape(-1),
                    # Retained so that the omitted-uncertainty diagnostic in
                    # StudyResult.coverage_inflated_mean_se can be computed
                    # after the fact, without refitting: log_mean is the
                    # ensemble-mean log effect and log_se_between the spread
                    # across independently initialised members at each point.
                    "log_mean": np.asarray(estimate.log_mean).reshape(-1),
                    "log_se_between": np.asarray(estimate.log_se).reshape(-1),
                }
                component = getattr(
                    extractor, "last_laplace_components", None
                )
                if component is not None:
                    components[name] = component
            results[target].replicates.append(
                ReplicateResult(
                    seed=seed,
                    estimates=estimates,
                    laplace_components=components or None,
                    fit_summary=trainer.fit_summary,
                )
            )
    return results


def summarise_regions(
    study: StudyResult,
    *,
    term: str = "x",
    interior=None,
    boundary=None,
) -> dict:
    """Return RMSE, squared bias, variance, and coverage by region."""
    output = {}
    for label, mask in (
        ("tot", None),
        ("int", interior),
        ("bnd", boundary),
    ):
        if label != "tot" and mask is None:
            continue
        output[f"err_{label}"], output[f"err_{label}_se"] = (
            study.rmse_mean_se(term, mask=mask)
        )
        output[f"bias2_{label}"], output[f"bias2_{label}_se"] = (
            study.bias2_mean_se(term, mask=mask)
        )
        output[f"var_{label}"], output[f"var_{label}_se"] = (
            study.variance_mean_se(term, mask=mask)
        )
        output[f"cov_{label}"], output[f"cov_{label}_se"] = (
            study.coverage_mean_se(term, mask=mask)
        )
        # Mean interval width on the logRR scale. Coverage alone cannot separate a
        # well-calibrated interval from a usefully uninformative one; width lets
        # calibration be read as coverage attained at a given width.
        output[f"width_{label}"] = study.width_mean(term, mask=mask)
        # Counterfactual coverage of the Laplace term alone. The reported
        # interval already carries the between-member variance, so this is what
        # coverage would be if the learned representation were treated as fixed;
        # the difference isolates what the representation term contributes.
        cond, cond_se = study.coverage_conditional_mean_se(term, mask=mask)
        if cond == cond:              # skip NaN when the inputs were not retained
            output[f"covcond_{label}"] = cond
            output[f"covcond_{label}_se"] = cond_se
    return output

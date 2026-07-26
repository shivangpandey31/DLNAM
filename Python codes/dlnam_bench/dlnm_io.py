"""Load R's exported DLNM effects into ``StudyResult`` objects.

The cumulative and surface loaders attach the analytic DGP truth to the same
manifest grid used by R. Consequently, DLNAM and DLNM-family estimators are
scored by the identical bias, variance, RMSE, and coverage implementation.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from dlnam import make_link
from dlnam.terms.base import Centering
from dlnam_sim.dgp import DataGeneratingProcess
from dlnam_sim.study import ReplicateResult, StudyResult


def _truth_on_grid(dgp, term, grid, centering, link):
    curve = dgp.truth_curve(term, grid, centering)
    if link.name in ("log", "logit"):
        return np.exp(curve.log_effect)
    return curve.log_effect


def _scoring_link(dgp):
    """The link the comparator estimates are scored on.

    Taken from the data-generating process rather than assumed, so the truth is
    built on the same scale the R estimators write out. The R fitting scripts
    report effects as relative risks (exp of the linear-predictor contrast), so a
    DGP on a link that is not log or logit would need those scripts changed too;
    that mismatch is caught here rather than silently mis-scaling the truth.
    """
    link = getattr(dgp, "link", None) or make_link("log")
    if link.name not in ("log", "logit"):
        raise ValueError(
            f"comparator outputs are exponentiated contrasts, which assumes a log "
            f"or logit link, but the DGP uses {link.name!r}; update the R fitting "
            f"scripts before scoring on this link"
        )
    return link


def _surface_truth_on_grid(dgp, term, grid, reference, link):
    """Return the centred true surface as ``(n_lags, n_grid)``."""
    true_term = dgp.true_terms[term]
    if true_term.kind != "surface":
        raise ValueError(f"{term!r} is not a surface term")
    lags = np.arange(true_term.lag_max + 1, dtype=float)
    raw = np.asarray(true_term.fn(grid[:, None], lags[None, :]), dtype=float)
    ref = np.asarray(
        true_term.fn(np.asarray([[reference]], dtype=float), lags[None, :]),
        dtype=float,
    )
    log_effect = (raw - ref).T
    return np.exp(log_effect) if link.name in ("log", "logit") else log_effect


def _surface_column(df, column, grid, lags):
    """Read one surface column on the manifest grid.

    Duplicate ``(lag, value)`` rows are averaged. This accommodates posterior
    summaries that repeat identical prediction locations while preserving the
    common scoring grid used by every estimator.
    """
    if column not in df:
        raise ValueError(f"surface output is missing required column {column!r}")
    grouped = (
        df[["lag", "value", column]]
        .groupby(["lag", "value"], as_index=False, sort=True)
        .mean(numeric_only=True)
    )
    rows = []
    for lag in lags:
        sub = grouped[np.isclose(grouped["lag"].to_numpy(float), lag)]
        if sub.empty:
            raise ValueError(f"surface output has no predictions for lag {lag:g}")
        x = sub["value"].to_numpy(float)
        y = sub[column].to_numpy(float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        if len(x) == len(grid) and np.allclose(x, grid, rtol=0.0, atol=1e-8):
            rows.append(y)
            continue
        if x[0] > grid[0] + 1e-8 or x[-1] < grid[-1] - 1e-8:
            raise ValueError(
                f"surface predictions for lag {lag:g} do not span the manifest grid"
            )
        rows.append(np.interp(grid, x, y))
    return np.asarray(rows, dtype=float)


def load_dlnm_study(
    out_dir: str,
    dgp: DataGeneratingProcess,
    scenario: str,
    centering: Centering,
    term: str = "x",
    prefix: str = "",
) -> StudyResult:
    """Assemble cumulative curves for one estimator and scenario."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    grid = np.asarray(manifest["grid"], dtype=float)
    link = _scoring_link(dgp)
    truth = _truth_on_grid(dgp, term, grid, centering, link)

    def _pre(path):
        directory, basename = os.path.split(path)
        return os.path.join(directory, prefix + basename)

    result = StudyResult(truth={term: truth}, grids={term: grid})
    for record in manifest["datasets"]:
        if record["scenario"] != scenario:
            continue
        path = os.path.join(out_dir, _pre(record["cumulative"]))
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        estimates = {
            term: {
                "mean": frame["fit"].values,
                "lo": frame["lo"].values,
                "hi": frame["hi"].values,
            }
        }
        result.replicates.append(
            ReplicateResult(seed=record["rep"], estimates=estimates)
        )
    return result


def load_dlnm_surface_study(
    out_dir: str,
    dgp: DataGeneratingProcess,
    scenario: str,
    centering: Centering,
    term: str = "x",
    prefix: str = "",
) -> StudyResult:
    """Assemble complete exposure-lag surfaces for one estimator and scenario.

    Surface arrays are flattened in lag-major order. The associated grid is the
    exposure grid repeated once per lag, allowing cumulative-curve
    interior/boundary masks to be repeated over all lags.
    """
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    grid = np.asarray(manifest["grid"], dtype=float)
    reference = float(manifest["reference"])
    lags = np.arange(int(manifest["lag_max"]) + 1, dtype=float)
    link = _scoring_link(dgp)
    truth = _surface_truth_on_grid(dgp, term, grid, reference, link).reshape(-1)
    surface_grid = np.tile(grid, len(lags))

    def _pre(path):
        directory, basename = os.path.split(path)
        return os.path.join(directory, prefix + basename)

    result = StudyResult(truth={term: truth}, grids={term: surface_grid})
    for record in manifest["datasets"]:
        if record["scenario"] != scenario:
            continue
        path = os.path.join(out_dir, _pre(record["surface"]))
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        estimates = {
            term: {
                "mean": _surface_column(frame, "rr", grid, lags).reshape(-1),
                "lo": _surface_column(frame, "lo", grid, lags).reshape(-1),
                "hi": _surface_column(frame, "hi", grid, lags).reshape(-1),
            }
        }
        result.replicates.append(
            ReplicateResult(seed=record["rep"], estimates=estimates)
        )
    return result

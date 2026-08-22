"""
scenarios_joint.py -- joint four-exposure DGP for the joint MC.

This deliberately reuses the single-exposure simulation surfaces from
dlnam_sim.scenarios, but treats them as four concurrent exposures in one
Poisson time-series DGP. The exposure processes share the same marginal
structure as the main MC, with phase shifts and correlated AR(1) innovations so
the joint fit has realistic confounding without adding a new data mechanism.
"""
from __future__ import annotations

import numpy as np

from dlnam.links import make_link
from dlnam_sim.dgp import DataGeneratingProcess, FunctionTerm, PoissonSampler
from dlnam_sim.scenarios import (
    AR_EPS_SD,
    AR_RHO,
    EXPOSURE_AMPLITUDE,
    EXPOSURE_MEAN,
    EXPOSURE_PERIOD,
    INTERCEPT,
    LAG_MAX,
    REFERENCE,
    SURFACE_FUNCTIONS,
    VALUE_RANGE,
)


EXPOSURES = ("dgp1", "dgp2", "dgp3", "dgp4")
PHASES = {
    "dgp1": 0.0,
    "dgp2": np.pi / 6.0,
    "dgp3": np.pi / 3.0,
    "dgp4": np.pi / 2.0,
}

EFFECT_SCALE = 1.0
EXPOSURE_CORR = 0.5


def _scaled_surface(fn, scale):
    def wrapped(v, lag):
        return scale * fn(v, lag)
    return wrapped


def _zero_surface(v, lag):
    return 0.0 * v


def _joint_weather_matrix(T, rng, names, *, corr=EXPOSURE_CORR):
    """Return T x K correlated seasonal AR(1) exposures."""
    names = tuple(names)
    k = len(names)
    t = np.arange(T, dtype=float)[:, None]
    phases = np.asarray([PHASES.get(name, 0.0) for name in names], dtype=float)[None, :]
    seasonal = EXPOSURE_MEAN + EXPOSURE_AMPLITUDE * np.sin(
        2.0 * np.pi * t / EXPOSURE_PERIOD + phases
    )

    corr_mat = np.full((k, k), float(corr), dtype=float)
    np.fill_diagonal(corr_mat, 1.0)
    innov_cov = (AR_EPS_SD ** 2) * corr_mat

    u = np.zeros((T, k), dtype=float)
    stat_cov = innov_cov / max(1.0 - AR_RHO ** 2, 1e-12)
    u[0] = rng.multivariate_normal(np.zeros(k), stat_cov)
    eps = rng.multivariate_normal(np.zeros(k), innov_cov, size=T)
    for i in range(1, T):
        u[i] = AR_RHO * u[i - 1] + eps[i]
    return seasonal + u


def correlated_weather_samplers(names=EXPOSURES, *, corr=EXPOSURE_CORR):
    """Create one sampler per exposure, backed by one shared correlated draw.

    DataGeneratingProcess asks each sampler for one column at a time. The closure
    below generates the full multivariate exposure matrix on the first request
    for a simulation and serves the remaining columns from the cache.
    """
    names = tuple(names)
    cache = {"T": None, "values": None, "remaining": set()}

    def ensure(T, rng):
        if cache["values"] is None or cache["T"] != T or not cache["remaining"]:
            mat = _joint_weather_matrix(T, rng, names, corr=corr)
            cache["T"] = T
            cache["values"] = {name: mat[:, i].copy() for i, name in enumerate(names)}
            cache["remaining"] = set(names)

    def make_sampler(name):
        def sampler(T, rng):
            ensure(T, rng)
            out = cache["values"][name].copy()
            cache["remaining"].discard(name)
            if not cache["remaining"]:
                cache["values"] = None
            return out
        return sampler

    return {name: make_sampler(name) for name in names}


def joint_dgp(
    lag_max=LAG_MAX,
    *,
    effect_scale=EFFECT_SCALE,
    exposure_corr=EXPOSURE_CORR,
    include_null=False,
):
    names = list(EXPOSURES)
    if include_null:
        names.append("null")

    true_terms = {}
    for name in names:
        fn = (
            _zero_surface
            if name == "null"
            else _scaled_surface(SURFACE_FUNCTIONS[name], effect_scale)
        )
        true_terms[name] = FunctionTerm(
            name,
            fn,
            kind="surface",
            lag_max=lag_max,
            value_range=VALUE_RANGE,
        )

    return DataGeneratingProcess(
        true_terms=true_terms,
        covariate_sampler=correlated_weather_samplers(names, corr=exposure_corr),
        intercept=INTERCEPT,
        link=make_link("log"),
        sampler=PoissonSampler(),
        target_col="death",
    )


__all__ = [
    "EXPOSURES",
    "PHASES",
    "EFFECT_SCALE",
    "EXPOSURE_CORR",
    "INTERCEPT",
    "LAG_MAX",
    "REFERENCE",
    "VALUE_RANGE",
    "joint_dgp",
]

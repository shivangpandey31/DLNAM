"""Data-generating processes for the DLNAM-vs-DLNM simulation study."""

from __future__ import annotations

import numpy as np
import torch

from dlnam.links import make_link
from dlnam_sim.dgp import DataGeneratingProcess, FunctionTerm, PoissonSampler


LAG_MAX = 14
REFERENCE = 20.0
VALUE_RANGE = (0.0, 40.0)
INTERCEPT = float(np.log(50.0))

EXPOSURE_MEAN = 20.0
EXPOSURE_AMPLITUDE = 12.0
EXPOSURE_PERIOD = 365.0
AR_RHO = 0.8
AR_EPS_SD = 2.0


def _is_t(x):
    return hasattr(x, "detach")


def pexp(x):
    return torch.exp(x) if _is_t(x) else np.exp(x)


def sig(x):
    return 1.0 / (1.0 + pexp(-x))


def _sum_last(x):
    return (
        x.sum(dim=-1, keepdim=True)
        if _is_t(x)
        else x.sum(axis=-1, keepdims=True)
    )


def bump(x, center, width):
    return pexp(-0.5 * ((x - center) / width) ** 2)


def norm_decay(lag, scale):
    w = pexp(-lag / scale)
    return w / (_sum_last(w) + 1e-12)


def norm_bump(lag, center, width):
    w = bump(lag, center, width)
    return w / (_sum_last(w) + 1e-12)


def f_dgp1(v, lag):
    """DGP 1."""
    cold = 0.50 * sig((15.0 - v) / 2.5)
    heat = 0.60 * sig((v - 25.0) / 1.5)
    return (cold + heat) * norm_decay(lag, 3.0)


def f_dgp2(v, lag):
    """DGP 2."""
    high = sig((v - 30.0) / 1.0)
    low = sig((12.0 - v) / 3.0)
    early = norm_bump(lag, 1.0, 1.5)
    late = norm_bump(lag, 9.0, 2.0)
    trough = norm_bump(lag, 5.0, 1.5)
    heat = 0.70 * high * early
    cold = low * (0.70 * late - 0.35 * trough)
    return heat + cold


def f_dgp3(v, lag):
    """DGP 3."""
    heat = 0.70 * bump(v, 32.0, 2.0) * norm_bump(lag, 1.0, 1.5)
    cold = 0.40 * sig((10.0 - v) / 3.0) * norm_decay(lag, 4.0)
    return heat + cold


def f_dgp4(v, lag):
    """DGP 4."""
    heat = 0.70 * sig((v - 28.0) / 1.0) * norm_bump(lag, 1.0, 1.0)
    cold = 0.60 * sig((12.0 - v) / 1.0) * norm_bump(lag, 9.0, 1.5)
    return heat + cold


def gp_weather(T, rng):
    t = np.arange(T)
    seasonal = EXPOSURE_MEAN + EXPOSURE_AMPLITUDE * np.sin(
        2.0 * np.pi * t / EXPOSURE_PERIOD
    )
    u = np.zeros(T, dtype=float)
    stationary_sd = AR_EPS_SD / np.sqrt(max(1.0 - AR_RHO ** 2, 1e-12))
    u[0] = rng.normal(0.0, stationary_sd)
    eps = rng.normal(0.0, AR_EPS_SD, T)
    for i in range(1, T):
        u[i] = AR_RHO * u[i - 1] + eps[i]
    return seasonal + u


SCENARIO_KEYS = ("dgp1", "dgp2", "dgp3", "dgp4")

SCENARIO_DISPLAY_NAMES = {
    "dgp1": "DGP 1",
    "dgp2": "DGP 2",
    "dgp3": "DGP 3",
    "dgp4": "DGP 4",
}

LEGACY_SCENARIO_ALIASES = {
    "smooth": "dgp1",
    "delayed_peaks": "dgp2",
    "localized_peak": "dgp3",
    "tilting_threshold": "dgp4",
}

SURFACE_FUNCTIONS = {
    "dgp1": f_dgp1,
    "dgp2": f_dgp2,
    "dgp3": f_dgp3,
    "dgp4": f_dgp4,
}


def canonical_scenario(name: str) -> str:
    """Return the canonical DGP key, accepting legacy result-file names."""
    return LEGACY_SCENARIO_ALIASES.get(name, name)


def _build(name, lag_max=LAG_MAX):
    name = canonical_scenario(name)
    if name not in SURFACE_FUNCTIONS:
        valid = ", ".join(SURFACE_FUNCTIONS)
        raise KeyError(f"unknown simulation scenario {name!r}; expected one of: {valid}")
    return DataGeneratingProcess(
        true_terms={
            "x": FunctionTerm(
                "x",
                SURFACE_FUNCTIONS[name],
                kind="surface",
                lag_max=lag_max,
                value_range=VALUE_RANGE,
            )
        },
        covariate_sampler={"x": gp_weather},
        intercept=INTERCEPT,
        link=make_link("log"),
        sampler=PoissonSampler(),
        target_col="death",
    )


def scenarios(lag_max=LAG_MAX) -> dict:
    return {name: _build(name, lag_max) for name in SCENARIO_KEYS}


__all__ = [
    "AR_EPS_SD",
    "AR_RHO",
    "EXPOSURE_AMPLITUDE",
    "EXPOSURE_MEAN",
    "EXPOSURE_PERIOD",
    "INTERCEPT",
    "LAG_MAX",
    "REFERENCE",
    "SCENARIO_KEYS",
    "SCENARIO_DISPLAY_NAMES",
    "LEGACY_SCENARIO_ALIASES",
    "SURFACE_FUNCTIONS",
    "canonical_scenario",
    "VALUE_RANGE",
    "f_dgp1",
    "f_dgp2",
    "f_dgp3",
    "f_dgp4",
    "gp_weather",
    "scenarios",
]

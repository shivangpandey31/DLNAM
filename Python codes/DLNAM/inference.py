"""
inference.py — the ONE place effects and intervals are computed.

Replaces the duplicated CI math currently living in both evaluation.py and
visualization.py. Both the evaluator and the visualizer consume `EffectEstimate`
from `EffectExtractor`; neither reimplements centering or interval logic.

Uncertainty methods are split by purpose, per your decision:
  WaldUQ      — the coverage-bearing interval. This is the one the MC study
                evaluates for nominal coverage. Available for every
                distribution, hence the default for reported inference.
  EnsembleUQ  — DISPLAY ONLY. The spread across ensemble members; communicates
                optimisation/epistemic stability, makes NO coverage claim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .links import Link
from .terms.base import Centering, EffectCurve


@dataclass
class EffectEstimate:
    """A term's effect with uncertainty, on the response scale (RR/OR).

    The shared output of inference. Ground truth is representable as an
    EffectEstimate with lo=hi=None, so truth and estimate overlay trivially and
    coverage is `lo <= truth <= hi` pointwise.
    """
    name: str
    grid_raw: np.ndarray
    mean: np.ndarray                 # response scale (e.g. RR)
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    log_mean: Optional[np.ndarray] = None
    log_se: Optional[np.ndarray] = None
    per_lag: Optional[np.ndarray] = None
    label: Optional[str] = None
    ci_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Uncertainty methods
# ---------------------------------------------------------------------------

class UncertaintyMethod(ABC):
    @abstractmethod
    def interval(self,
                 log_effect_per_member: np.ndarray,   # (n_members, G)
                 *,
                 link: Link,
                 alpha: float,
                 phi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) on the response scale at the given grid points."""


class WaldUQ(UncertaintyMethod):
    """Coverage-bearing Wald interval on the effect curve.

    OPEN DESIGN DECISION — flagged for your call, because the coverage you'll
    measure depends entirely on it: an NN ensemble has no Fisher information by
    default, so the standard error of the centered log-effect must come from
    one of:
        (a) ensemble spread reused as an SE estimate (cheapest; what the old
            code effectively did via sd_rr = mean_rr * sd_log),
        (b) a delta-method propagation from a parameter-covariance estimate,
        (c) a bootstrap over refits,
        (d) a last-layer Laplace / sandwich estimator.
    Whichever you choose, it lives HERE and only here. The MC harness will tell
    you which actually attains nominal coverage — that comparison is arguably
    the headline result, so keep this swappable.

    phi carries the quasi-likelihood overdispersion factor for the Poisson case.
    """
    def __init__(self, se_source: str = "ensemble"):
        self.se_source = se_source

    def interval(self, log_effect_per_member, *, link, alpha, phi=1.0):
        from scipy import stats
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        log_mean = np.mean(log_effect_per_member, axis=0)
        if self.se_source == "ensemble":
            se_log = np.std(log_effect_per_member, axis=0)
        else:
            raise NotImplementedError(
                f"Wald se_source='{self.se_source}' not implemented yet "
                "(bootstrap / laplace are future options)"
            )
        # Build the interval on the LINK scale, then map to the response scale,
        # so the RR/OR interval is correctly asymmetric.
        lo = _effect_to_response(link, log_mean - z * se_log)
        hi = _effect_to_response(link, log_mean + z * se_log)
        return lo, hi


class EnsembleUQ(UncertaintyMethod):
    """Display-only band from across-member spread. No coverage claim."""
    def interval(self, log_effect_per_member, *, link, alpha, phi=1.0):
        from scipy import stats
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        log_mean = np.mean(log_effect_per_member, axis=0)
        sd_log = np.std(log_effect_per_member, axis=0)
        mean_resp = _effect_to_response(link, log_mean)
        sd_resp = mean_resp * sd_log                      # delta method
        return mean_resp - z * sd_resp, mean_resp + z * sd_resp


# ---------------------------------------------------------------------------
# Effect extraction — used by BOTH evaluate.py and visualize.py
# ---------------------------------------------------------------------------

class EffectExtractor:
    """Pulls a centered, response-scale effect (+ interval) for any term out of
    a trained ensemble. The visualizer plots it; the evaluator scores it; the
    MC study compares it to truth. One implementation, no duplication."""

    def __init__(self,
                 ensemble: Sequence,          # list[DLNAM]
                 link: Link,
                 uq: UncertaintyMethod,
                 centering: Centering,
                 phi: float = 1.0):
        self.ensemble = ensemble
        self.link = link
        self.uq = uq
        self.centering = centering
        self.phi = phi

    def extract(self,
                term_name: str,
                grid_raw: Optional[np.ndarray] = None,
                alpha: float = 0.05) -> EffectEstimate:
        """For each ensemble member: get the term's centered log-effect on the
        grid; stack -> (n_members, G); mean -> response scale; delegate the band
        to self.uq. Returns an EffectEstimate."""
        if grid_raw is None:
            grid_raw = self.ensemble[0].term(term_name).default_grid()
        grid_raw = np.asarray(grid_raw, dtype=float)

        per_member = []
        label = None
        for model in self.ensemble:
            curve = model.term(term_name).effect(grid_raw, self.centering)
            per_member.append(curve.log_effect)
            label = curve.label
        per_member = np.asarray(per_member)               # (M, G)

        log_mean = np.mean(per_member, axis=0)
        mean_resp = _effect_to_response(self.link, log_mean)
        lo, hi = self.uq.interval(
            per_member, link=self.link, alpha=alpha, phi=self.phi
        )
        return EffectEstimate(
            name=term_name, grid_raw=grid_raw, mean=mean_resp,
            lo=lo, hi=hi, log_mean=log_mean,
            log_se=np.std(per_member, axis=0),
            label=label, ci_label=type(self.uq).__name__,
        )

    @staticmethod
    def from_truth(curve: EffectCurve, link: Link) -> EffectEstimate:
        """Wrap a ground-truth EffectCurve as an EffectEstimate (no interval),
        so truth and estimate are the same type downstream."""
        return EffectEstimate(
            name=curve.name,
            grid_raw=np.asarray(curve.grid_raw, dtype=float),
            mean=_effect_to_response(link, np.asarray(curve.log_effect)),
            log_mean=np.asarray(curve.log_effect, dtype=float),
            label=curve.label,
        )


def _effect_to_response(link: Link, effect: np.ndarray) -> np.ndarray:
    """NumPy bridge to Link.effect_to_response (which is defined on tensors)."""
    if link.name in ("log", "logit"):
        return np.exp(effect)
    return effect

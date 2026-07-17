"""
inference.py - effect extraction and uncertainty intervals.

Both the evaluator and the visualizer consume `EffectEstimate` from
`EffectExtractor`; neither reimplements centering or interval logic.

Supported interval types:
  confidence -- pointwise effect confidence interval from last-layer Laplace SEs;
  prediction -- pointwise prediction interval from Poisson/quasi-Poisson outcome
                variance;
  ensemble   -- pointwise interval from ensemble spread, useful as a diagnostic
                but not as a sampling-coverage statement.
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
                 log_effect_per_member: np.ndarray,   # (n_members, ...)
                 *,
                 link: Link,
                 alpha: float,
                 phi: float = 1.0,
                 mean_count: Optional[np.ndarray] = None,
                 laplace_se: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) on the response scale at the given grid points."""


class IntervalUQ(UncertaintyMethod):
    """Coverage-bearing Wald interval on the effect curve.

    The class centralises the normal-approximation interval form. The selected
    interval type determines where the log-scale standard error comes from.
    """
    _ALIASES = {
        "confidence": "laplace",
        "effect": "laplace",
        "laplace": "laplace",
        "prediction": "poisson",
        "predictive": "poisson",
        "poisson": "poisson",
        "ensemble": "ensemble",
    }

    def __init__(self, interval: str = "ensemble"):
        if interval not in self._ALIASES:
            valid = ", ".join(sorted(self._ALIASES))
            raise ValueError(f"unknown interval='{interval}' (valid: {valid})")
        self.interval_type = interval
        self.se_source = self._ALIASES[interval]

    @property
    def label(self) -> str:
        if self.se_source == "laplace":
            return "pointwise confidence interval"
        if self.se_source == "poisson":
            return "pointwise prediction interval"
        return "ensemble spread interval"

    def interval(self, log_effect_per_member, *, link, alpha, phi=1.0,
                 mean_count=None, laplace_se=None):
        """Interval on the centered effect curve.

        se_source:
          'ensemble' -- band from the spread (std) of the centered log-effect
              ACROSS ensemble members. This reflects init/optimisation disagreement,
              NOT estimator sampling uncertainty; reported as a diagnostic.
          'poisson'  -- Poisson interval using Var = mean: at each grid point the
              predicted COUNT mu = mean_count is treated as Poisson(mu) (Var = mu),
              giving a log-rate SE of sqrt(phi/mu) by the delta method; the interval
              is mean +/- z*SE on the log scale, mapped to the effect (RR) scale.
              For the large counts here (mu ~ 1e2) this normal/Var=mean form and the
              exact Poisson quantiles coincide. phi scales the variance for
              quasi-Poisson overdispersion (Var = phi*mu). Requires mean_count.
          'laplace'  -- last-layer Laplace CONFIDENCE interval: SE on the log scale
              from propagation through the Poisson last-layer Fisher information
              (see dlnam.laplace.LastLayerLaplace). This targets the TRUE EFFECT CURVE
              (like the cross-basis CI), unlike 'poisson' which is a prediction
              interval for data. Requires laplace_se, precomputed by the extractor.
        """
        from scipy import stats
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        log_mean = np.mean(log_effect_per_member, axis=0)

        if self.se_source == "ensemble":
            se_log = np.std(log_effect_per_member, axis=0)
            lo = _effect_to_response(link, log_mean - z * se_log)
            hi = _effect_to_response(link, log_mean + z * se_log)
            return lo, hi

        if self.se_source == "poisson":
            if mean_count is None:
                raise ValueError("se_source='poisson' needs mean_count "
                                 "(predicted counts per grid point)")
            mu = np.asarray(mean_count, dtype=float)
            mu = np.clip(mu, 1e-9, None)
            # Var = phi * mu (Poisson, quasi-Poisson if phi>1). SE on the COUNT
            # scale; the predicted-rate log-SE is then sqrt(phi/mu) by the delta
            # method (d log mu = d mu / mu, Var(d mu)=phi*mu -> Var(log mu)=phi/mu).
            se_log = np.sqrt(phi / mu)
            lo = _effect_to_response(link, log_mean - z * se_log)
            hi = _effect_to_response(link, log_mean + z * se_log)
            return lo, hi

        if self.se_source == "laplace":
            if laplace_se is None:
                raise ValueError("se_source='laplace' needs laplace_se "
                                 "(precomputed per-grid-point SE on the log scale)")
            se_log = np.asarray(laplace_se, dtype=float)
            lo = _effect_to_response(link, log_mean - z * se_log)
            hi = _effect_to_response(link, log_mean + z * se_log)
            return lo, hi

        raise NotImplementedError(
            f"se_source='{self.se_source}' not implemented "
            "(implemented: 'ensemble', 'poisson', 'laplace')"
        )


class ConfidenceIntervalUQ(IntervalUQ):
    """Pointwise effect confidence interval from last-layer Laplace SEs."""
    def __init__(self):
        super().__init__("confidence")


class PredictionIntervalUQ(IntervalUQ):
    """Pointwise prediction interval using Poisson/quasi-Poisson outcome variance."""
    def __init__(self):
        super().__init__("prediction")


class EnsembleIntervalUQ(IntervalUQ):
    """Pointwise interval from ensemble spread; diagnostic, not sampling coverage."""
    def __init__(self):
        super().__init__("ensemble")


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
                 phi: float = 1.0,
                 laplace_inputs: Optional[dict] = None):
        self.ensemble = ensemble
        self.link = link
        self.uq = uq
        self.centering = centering
        self.phi = phi
        # For se_source='laplace': per-member data needed to build the last-layer
        # Fisher information. dict with keys:
        #   'feat_windows' : list per member of (N, Lp1) scaled training windows
        #                    (or a single array reused for all members)
        #   'mu_hat'       : list per member of (N,) fitted counts (or single array)
        #   'ref'          : reference value for centering the estimand
        # If None, se_source='laplace' will raise when interval() is called.
        self.laplace_inputs = laplace_inputs

    @classmethod
    def with_laplace(cls, ensemble, prepared, link, centering):
        """Build an extractor configured for se_source='laplace', self-assembling
        everything the joint last-layer Laplace needs from the trained `ensemble`
        and the `prepared` training data (PreparedData). Runners call this instead
        of hand-plumbing windows/mu/phi.

        - per-member fitted counts mu_hat are computed by a forward pass
        - phi (quasi-Poisson dispersion) is estimated by the mean Pearson statistic
          across members, so the DLNAM interval carries the SAME dispersion notion
          as the classical DLNM (fair like-for-like coverage)
        - the full prepared input dict is passed (all terms), so the covariance is
          JOINT across every term and includes the intercept;
        - subnet mixing weights are conditioned on at their fitted values.
        """
        import torch as _t
        dev = next(ensemble[0].parameters()).device
        inp = {k: v.to(dev) for k, v in prepared.inputs.items()}
        y = np.asarray(prepared.y).reshape(-1)
        mus, phis = [], []
        with _t.no_grad():
            for m in ensemble:
                mu = m(inp).detach().cpu().numpy().reshape(-1).clip(1e-9)
                mus.append(mu)
                dof = max(len(y) - 1, 1)
                phis.append(float(np.sum((y - mu) ** 2 / mu) / dof))
        phi = float(np.mean(phis)) if phis else 1.0
        phi = max(phi, 1.0)   # never below Poisson
        li = {
            "prepared_inputs": [inp] * len(ensemble),
            "mu_hat": mus,
        }
        return cls(ensemble, link, IntervalUQ("laplace"), centering,
                   phi=phi, laplace_inputs=li)

    def _laplace_reference(self, term):
        if self.centering.method in ("reference", "custom"):
            if self.centering.value is None:
                raise ValueError(f"centering='{self.centering.method}' needs value")
            return float(self.centering.value)
        if self.centering.method == "median":
            if getattr(term, "_data_median", None) is None:
                raise ValueError("centering='median' needs a fitted term median")
            return float(term._data_median)
        raise NotImplementedError(
            "Laplace CIs currently support reference/custom/median centering; "
            "mean centering would require a training-distribution contrast."
        )

    def _laplace_se(self, term_name, grid_raw, ref, per_member, *, surface=False):
        """Laplace SE for either a cumulative/1D effect or a 2D surface."""
        if self.laplace_inputs is None:
            raise ValueError("se_source='laplace' requires laplace_inputs on "
                             "the EffectExtractor (prepared_inputs, mu_hat)")
        from .laplace import LastLayerLaplace, pooled_evidence_lambda
        li = self.laplace_inputs
        prepared_inputs = li["prepared_inputs"]     # per-member dict or single
        mu = li["mu_hat"]                           # per-member (N,) or single
        M = len(self.ensemble)

        def _lap(i):
            pin = prepared_inputs[i] if isinstance(prepared_inputs, (list, tuple)) \
                else prepared_inputs
            m = mu[i] if isinstance(mu, (list, tuple)) else mu
            return LastLayerLaplace(self.ensemble[i], pin, m, phi=self.phi)

        laps = [_lap(i) for i in range(M)]
        # ONE lambda for the whole ensemble (= one estimator). It is re-estimated
        # inside each MC replicate from the fitted model, so replicates remain
        # independent.
        shared_lambda = pooled_evidence_lambda([lap.evidence_terms() for lap in laps])
        self.last_prior_precision = shared_lambda

        for lap in laps:
            lap.set_prior_precision(shared_lambda)

        if surface:
            within_vars = np.array([lap.surface_se(term_name, grid_raw, ref) ** 2
                                    for lap in laps])
        else:
            within_vars = np.array([lap.effect_se(term_name, grid_raw, ref) ** 2
                                    for lap in laps])
        var_within = within_vars.mean(axis=0)
        # The reported estimand is the ensemble-mean effect. The Laplace CI uses
        # within-member sampling uncertainty averaged across ensemble members.
        laplace_se = np.sqrt(var_within)
        self.last_laplace_components = {"se_total": laplace_se}
        return laplace_se

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
        # predicted COUNT per grid point = exp(intercept + centered log-effect),
        # using the ensemble-mean intercept; needed by se_source='poisson'
        # (Var=mean). Harmless for se_source='ensemble' (ignored there).
        try:
            icpt = float(np.mean([float(m.intercept.detach().cpu())
                                  for m in self.ensemble]))
            mean_count = np.exp(icpt + log_mean)
        except Exception:
            mean_count = None
        # se_source='laplace': JOINT last-layer Laplace (all terms + intercept),
        # computed per member and averaged as within-member sampling variance.
        laplace_se = None
        if getattr(self.uq, "se_source", None) == "laplace":
            ref = self._laplace_reference(self.ensemble[0].term(term_name))
            laplace_se = self._laplace_se(term_name, grid_raw, ref, per_member)

        lo, hi = self.uq.interval(
            per_member, link=self.link, alpha=alpha, phi=self.phi,
            mean_count=mean_count, laplace_se=laplace_se,
        )
        return EffectEstimate(
            name=term_name, grid_raw=grid_raw, mean=mean_resp,
            lo=lo, hi=hi, log_mean=log_mean,
            log_se=np.std(per_member, axis=0),
            label=label, ci_label=getattr(self.uq, "label", type(self.uq).__name__),
        )

    def extract_surface(self,
                        term_name: str,
                        grid_raw: Optional[np.ndarray] = None,
                        alpha: float = 0.05) -> EffectEstimate:
        """Pointwise value-by-lag surface effect with matching Laplace CI.

        The estimand is f(x, lag) - f(ref, lag), returned on the response scale
        with arrays shaped (n_lags, n_grid). This is distinct from `extract`,
        which returns the cumulative-over-lag curve for a surface term.
        """
        term0 = self.ensemble[0].term(term_name)
        if not hasattr(term0, "per_lag_log_rr"):
            raise ValueError("extract_surface is only defined for surface terms")
        if grid_raw is None:
            grid_raw = term0.default_grid()
        grid_raw = np.asarray(grid_raw, dtype=float)
        ref = self._laplace_reference(term0)

        per_member = np.asarray([
            model.term(term_name).per_lag_log_rr(grid_raw, ref)
            for model in self.ensemble
        ])                                                   # (M, n_lags, G)
        log_mean = np.mean(per_member, axis=0)
        mean_resp = _effect_to_response(self.link, log_mean)

        laplace_se = None
        if getattr(self.uq, "se_source", None) == "laplace":
            laplace_se = self._laplace_se(term_name, grid_raw, ref, per_member,
                                          surface=True)

        lo, hi = self.uq.interval(
            per_member, link=self.link, alpha=alpha, phi=self.phi,
            mean_count=None, laplace_se=laplace_se,
        )
        return EffectEstimate(
            name=term_name, grid_raw=grid_raw, mean=mean_resp,
            lo=lo, hi=hi, log_mean=log_mean,
            log_se=np.std(per_member, axis=0),
            per_lag=mean_resp,
            label=getattr(term0, "name", term_name),
            ci_label=getattr(self.uq, "label", type(self.uq).__name__),
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

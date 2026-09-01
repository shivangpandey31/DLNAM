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


#: se_source values that require the last-layer Laplace machinery to be built.
LAPLACE_SE_SOURCES = ("laplace", "laplace_ensemble")


def needs_laplace(se_source) -> bool:
    """Whether this interval type needs `EffectExtractor.with_laplace`.

    Accepts either a resolved se_source or any user-facing alias, because
    callers reach this from both sides: `IntervalUQ.se_source` is already
    resolved, whereas experiment runners pass the alias they were configured
    with ("laplace+ensemble"). Resolving here keeps the two in step, so an
    extractor is never built without the Laplace inputs an interval will ask for.
    """
    resolved = IntervalUQ._ALIASES.get(se_source, se_source)
    return resolved in LAPLACE_SE_SOURCES


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
        "laplace+ensemble": "laplace_ensemble",
        "laplace_ensemble": "laplace_ensemble",
        "combined": "laplace_ensemble",
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
        if self.se_source == "laplace_ensemble":
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

        if self.se_source in ("laplace", "laplace_ensemble"):
            if laplace_se is None:
                raise ValueError(f"se_source='{self.se_source}' needs laplace_se "
                                 "(precomputed per-grid-point SE on the log scale)")
            se_log = np.asarray(laplace_se, dtype=float)
            if self.se_source == "laplace_ensemble":
                # Add the within-member Laplace variance and the between-member
                # spread. Population (ddof=0) spread, matching the convention in
                # EffectEstimate.log_se.
                between = np.std(log_effect_per_member, axis=0)
                se_log = np.sqrt(se_log ** 2 + between ** 2)
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
        #   'information_weight': list per member of (N,) GLM working weights
        #   'mu_hat'            : optional fitted means retained for diagnostics
        # If None, se_source='laplace' will raise when interval() is called.
        self.laplace_inputs = laplace_inputs

    @classmethod
    def with_laplace(cls, ensemble, prepared, link, centering, laplace_terms=None,
                     interval="laplace"):
        """Build an extractor for a Laplace-based interval, self-assembling
        everything the joint last-layer Laplace needs from the trained `ensemble`
        and the `prepared` training data (PreparedData). Runners call this instead
        of hand-plumbing windows/mu/phi.

        - per-member GLM information weights are computed by a forward pass:
          w=mu for log/Poisson and w=mu(1-mu) for logit/Bernoulli;
        - phi (quasi-Poisson dispersion) is estimated by the mean Pearson statistic
          across members for log/Poisson and fixed at one for logit/Bernoulli;
        - the full prepared input dict is passed (all terms), so the covariance is
          JOINT across every term and includes the intercept;
        - if `laplace_terms` is supplied, nuisance terms not listed there are
          conditioned on at their fitted values;
        - subnet mixing weights are conditioned on at their fitted values.

        `interval` selects which Laplace-based interval to build: 'laplace' uses
        the within-member last-layer variance alone, 'laplace+ensemble' adds the
        between-member spread. It must be passed through rather than assumed,
        since both need the same Laplace machinery and differ only in the
        variance they report.
        """
        if not needs_laplace(interval):
            raise ValueError(
                f"with_laplace requires a Laplace-based interval, got "
                f"{interval!r}; use the EffectExtractor constructor instead"
            )
        import torch as _t
        dev = next(ensemble[0].parameters()).device
        inp = {k: v.to(dev) for k, v in prepared.inputs.items()}
        y = np.asarray(prepared.y).reshape(-1)
        information_weights, mus, phis = [], [], []
        with _t.no_grad():
            for m in ensemble:
                mu = m(inp).detach().cpu().numpy().reshape(-1).clip(1e-9)
                mus.append(mu)
                if link.name == "log":
                    information_weights.append(mu)
                    dof = max(len(y) - 1, 1)
                    phis.append(float(np.sum((y - mu) ** 2 / mu) / dof))
                elif link.name == "logit":
                    information_weights.append(np.clip(mu * (1.0 - mu), 1e-9, None))
                    phis.append(1.0)
                else:
                    raise NotImplementedError(
                        "last-layer Laplace currently supports log/Poisson and "
                        "logit/Bernoulli links"
                    )
        phi = float(np.mean(phis)) if phis else 1.0
        if link.name == "log":
            phi = max(phi, 1.0)   # never below Poisson
        li = {
            "prepared_inputs": [inp] * len(ensemble),
            "information_weight": information_weights,
            "mu_hat": mus,
            "laplace_terms": None if laplace_terms is None else tuple(laplace_terms),
        }
        return cls(ensemble, link, IntervalUQ(interval), centering,
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
                             "the EffectExtractor "
                             "(prepared_inputs, information_weight)")
        from .laplace import LastLayerLaplace, pooled_evidence_lambda
        li = self.laplace_inputs
        prepared_inputs = li["prepared_inputs"]     # per-member dict or single
        information_weight = li.get("information_weight", li.get("mu_hat"))
        include_terms = li.get("laplace_terms")
        M = len(self.ensemble)

        def _lap(i):
            pin = prepared_inputs[i] if isinstance(prepared_inputs, (list, tuple)) \
                else prepared_inputs
            w = (information_weight[i] if isinstance(information_weight, (list, tuple))
                 else information_weight)
            return LastLayerLaplace(
                self.ensemble[i], pin, w, phi=self.phi, include_terms=include_terms
            )

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
        if needs_laplace(getattr(self.uq, "se_source", None)):
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
                        alpha: float = 0.05,
                        n_lag_points: Optional[int] = None) -> EffectEstimate:
        """Pointwise value-by-lag surface effect with matching Laplace CI.

        The estimand is f(x, lag) - f(ref, lag), returned on the response scale
        with arrays shaped (n_lags, n_grid). This is distinct from `extract`,
        which returns the cumulative-over-lag curve for a surface term.

        `n_lag_points` evaluates the component on that many equally spaced
        scaled lags in [0, 1] instead of the observed lag grid. The lag is a
        continuous input to the component, so this is a model evaluation, not
        interpolation; it exists so that a short window renders as a surface
        rather than as a handful of ridges.
        """
        term0 = self.ensemble[0].term(term_name)
        if not hasattr(term0, "per_lag_log_rr"):
            raise ValueError("extract_surface is only defined for surface terms")
        if grid_raw is None:
            grid_raw = term0.default_grid()
        grid_raw = np.asarray(grid_raw, dtype=float)
        ref = self._laplace_reference(term0)

        lag_scaled = None
        if n_lag_points is not None:
            if needs_laplace(getattr(self.uq, "se_source", None)):
                # The last-layer design is built on the observed lag grid, so a
                # denser grid would not line up with it.
                raise ValueError(
                    "n_lag_points is not supported with a Laplace interval; "
                    "use an ensemble interval for densely evaluated surfaces")
            lag_scaled = np.linspace(0.0, 1.0, int(n_lag_points))

        per_member = np.asarray([
            model.term(term_name).per_lag_log_rr(grid_raw, ref, lag_scaled)
            for model in self.ensemble
        ])                                                   # (M, n_lags, G)
        log_mean = np.mean(per_member, axis=0)
        mean_resp = _effect_to_response(self.link, log_mean)

        laplace_se = None
        if needs_laplace(getattr(self.uq, "se_source", None)):
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

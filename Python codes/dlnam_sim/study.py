"""
dlnam_sim/study.py — Monte Carlo harness for bias & coverage.

Each replicate: simulate(seed) -> DataProcessor -> Trainer -> EffectExtractor,
then compare each term's centered effect to the DGP truth on a FIXED grid.
Aggregated into pointwise bias, bias^2, variance, RMSE, and empirical coverage of the
(coverage-bearing) Wald interval.

Imports from dlnam and dlnam_sim only; invisible to the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dlnam import Trainer, DataProcessor, EffectExtractor, IntervalUQ, make_link
from dlnam.config import ModelConfig, TrainConfig
from dlnam.terms.base import Centering

from .dgp import DataGeneratingProcess


@dataclass
class ReplicateResult:
    seed: int
    estimates: dict          # term -> dict(mean, lo, hi)  each (G,)
    laplace_components: dict = None   # optional: term -> {se_total}
    fit_summary: dict = None


@dataclass
class StudyResult:
    truth: dict              # term -> true response-scale curve (G,)
    grids: dict              # term -> grid (G,)
    replicates: list = field(default_factory=list)

    def _stack(self, term, key):
        return np.array([r.estimates[term][key] for r in self.replicates])

    # bias / bias^2 / variance / RMSE are computed on the logRR scale (the estimand scale:
    # the surface is additive in log-mortality, log link), so that the standard
    # decomposition MSE = bias^2 + variance holds on the SAME scale as the reported
    # error metric. Pointwise MSE = mean_r (log est - log truth)^2;
    # bias = mean_r (log est - log truth); variance = var_r (log est).
    def _logdev(self, term):
        # (R, G) per-replicate logRR deviation from truth
        return np.log(self._stack(term, "mean")) - np.log(self.truth[term])[None, :]

    def bias(self, term):     # (G,) pointwise; caller masks via indexing
        return self._logdev(term).mean(0)

    def bias2(self, term):
        return self.bias(term) ** 2

    def variance(self, term):
        return np.log(self._stack(term, "mean")).var(0)

    def rmse(self, term):
        d = self._logdev(term)
        return np.sqrt((d ** 2).mean(0))

    def coverage(self, term):
        lo, hi = self._stack(term, "lo"), self._stack(term, "hi")
        t = self.truth[term][None, :]
        return ((t >= lo) & (t <= hi)).mean(0)

    # --- analytical SEs across replicates (per grid point), aggregated by mask ---
    def bias2_mean_se(self, term, mask=None):
        """Mean squared bias over the selected grid on the logRR scale.

        This is the bias component in MSE = bias^2 + variance. The SE is a
        delete-one-replicate jackknife SE for the regional mean squared bias.
        """
        d = self._logdev(term)                                         # (R,G)
        R = d.shape[0]
        m = self._mask(term, mask)
        bias2 = float((d.mean(0)[m] ** 2).mean())
        if R <= 1:
            return bias2, 0.0
        sums = d.sum(0)
        loo = np.empty(R, dtype=float)
        for r in range(R):
            b = (sums - d[r]) / (R - 1)
            loo[r] = (b[m] ** 2).mean()
        se = np.sqrt((R - 1) / R * np.sum((loo - loo.mean()) ** 2))
        return bias2, float(se)

    def variance_mean_se(self, term, mask=None):
        """Mean Monte-Carlo variance over the selected grid on the logRR scale."""
        d = self._logdev(term)                                         # (R,G)
        R = d.shape[0]
        m = self._mask(term, mask)
        v = float(d[:, m].var(axis=0).mean())
        if R <= 1:
            return v, 0.0
        loo = np.empty(R, dtype=float)
        for r in range(R):
            loo[r] = np.delete(d, r, axis=0)[:, m].var(axis=0).mean()
        se = np.sqrt((R - 1) / R * np.sum((loo - loo.mean()) ** 2))
        return v, float(se)

    def coverage_mean_se(self, term, mask=None):
        # binomial SE of a proportion over R reps: sqrt(p(1-p)/R)
        p_pp = self.coverage(term)                                     # (G,)
        R = self._stack(term, "mean").shape[0]
        se_pp = np.sqrt(p_pp * (1 - p_pp) / R)
        m = np.ones(len(p_pp), bool) if mask is None else np.asarray(mask, bool)
        return float(p_pp[m].mean()), float(se_pp[m].mean())

    def width_mean(self, term, mask=None, log=True):
        """Mean interval WIDTH, averaged over grid and replicates. Diagnosable at
        ANY R (unlike coverage, which needs many reps): a band that is too wide
        over-covers, too narrow under-covers. log=True measures width on the logRR
        scale (log hi - log lo), the estimand scale; else raw RR width (hi - lo)."""
        lo, hi = self._stack(term, "lo"), self._stack(term, "hi")       # (R,G)
        w = (np.log(hi) - np.log(lo)) if log else (hi - lo)
        m = np.ones(w.shape[1], bool) if mask is None else np.asarray(mask, bool)
        return float(w[:, m].mean())

    def simultaneous_coverage(self, term, mask=None):
        """Fraction of REPLICATES whose band contains the WHOLE true curve at once
        (all grid points in the mask simultaneously). Stricter than pointwise; does
        not saturate at 1.0 as easily, and is the honest calibration statement for a
        curve. Returns (rate, binomial SE)."""
        lo, hi = self._stack(term, "lo"), self._stack(term, "hi")       # (R,G)
        t = self.truth[term][None, :]
        contained = (t >= lo) & (t <= hi)                              # (R,G)
        m = np.ones(contained.shape[1], bool) if mask is None else np.asarray(mask, bool)
        per_rep = contained[:, m].all(axis=1)                          # (R,)
        R = len(per_rep); p = float(per_rep.mean())
        return p, float(np.sqrt(p * (1 - p) / R)) if R > 0 else 0.0

    def _mask(self, term, mask):
        # mask: None (all grid points), or a boolean array over the term's grid.
        g = self.grids[term]
        return np.ones(len(g), dtype=bool) if mask is None else np.asarray(mask, bool)

    def error_per_rep(self, term, log=True, mask=None):
        """Per-replicate mean absolute error vs truth: (R,). log=True uses
        |log(est) - log(truth)|. mask restricts the grid (interior/boundary)."""
        est = self._stack(term, "mean")
        t = self.truth[term][None, :]
        d = np.abs(np.log(est) - np.log(t)) if log else np.abs(est - t)
        m = self._mask(term, mask)
        return d[:, m].mean(1)

    def error_mean_se(self, term, log=True, mask=None):
        """(mean, Monte-Carlo SE) of the per-replicate error across replicates,
        optionally restricted to a grid mask (interior/boundary)."""
        e = self.error_per_rep(term, log=log, mask=mask)
        R = len(e)
        return float(e.mean()), float(e.std(ddof=1) / np.sqrt(R)) if R > 1 else 0.0

    def rmse_per_rep(self, term, log=True, mask=None):
        """Per-replicate RMSE vs truth: (R,). log=True uses (log est - log truth)^2
        (the estimand scale); mask restricts the grid (interior/boundary). Squared
        error penalises the DLNM's large, localised failures (edge artefacts,
        phantom cells) more than MAE, and composes with bias/variance as
        MSE = bias^2 + variance on the logRR scale."""
        est = self._stack(term, "mean")
        t = self.truth[term][None, :]
        d = (np.log(est) - np.log(t)) if log else (est - t)
        m = self._mask(term, mask)
        return np.sqrt((d[:, m] ** 2).mean(1))

    def rmse_mean_se(self, term, log=True, mask=None):
        """Regional RMSE and delta-method MC-SE on the selected scale.

        The point estimate is sqrt(mean squared error over replicates and grid),
        so RMSE^2 matches the reported bias^2 + variance decomposition.
        """
        est = self._stack(term, "mean")
        t = self.truth[term][None, :]
        d = (np.log(est) - np.log(t)) if log else (est - t)
        m = self._mask(term, mask)
        mse_rep = (d[:, m] ** 2).mean(1)
        mse = float(mse_rep.mean())
        rmse = float(np.sqrt(mse))
        if len(mse_rep) <= 1 or rmse == 0.0:
            return rmse, 0.0
        mse_se = float(mse_rep.std(ddof=1) / np.sqrt(len(mse_rep)))
        return rmse, mse_se / (2.0 * rmse)

    def boundary_mask(self, term, exposure_values, q=0.05):
        """Boolean grid mask: True where the grid value is in the sparse tails
        (below the q or above the 1-q quantile of the sampled exposure)."""
        g = self.grids[term]
        lo, hi = np.quantile(exposure_values, [q, 1 - q])
        return (g < lo) | (g > hi)

    def summary(self) -> dict:
        return {term: {"mean_bias2": float(self.bias2(term).mean()),
                       "mean_rmse": float(self.rmse(term).mean()),
                       "mean_coverage": float(self.coverage(term).mean())}
                for term in self.truth}

    def timing_summary(self) -> dict:
        """Aggregate Trainer.fit_summary across replicates."""
        times = [
            float(r.fit_summary["fit_seconds"])
            for r in self.replicates
            if getattr(r, "fit_summary", None) and "fit_seconds" in r.fit_summary
        ]
        if not times:
            return {}
        return {
            "fit_seconds_mean": float(np.mean(times)),
            "fit_seconds_sd": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "fit_seconds_total": float(np.sum(times)),
            "n_replicates": int(len(times)),
        }


@dataclass
class MonteCarloStudy:
    dgp: DataGeneratingProcess
    model_config: ModelConfig
    train_config: TrainConfig
    centering: Centering
    n_reps: int = 500
    n_obs: int = 5000
    alpha: float = 0.05
    base_seed: int = 0
    se_source: str = "prediction"
    device: Optional[str] = "cpu"

    def _grids_and_truth(self):
        grids, truth = {}, {}
        link = make_link(self.model_config.link)
        for name in self.model_config.terms:
            if name not in self.dgp.true_terms:
                continue
            grid = self.dgp.true_terms[name].default_grid()
            curve = self.dgp.truth_curve(name, grid, self.centering)
            grids[name] = grid
            truth[name] = np.exp(curve.log_effect) if link.name in ("log", "logit") \
                else curve.log_effect
        return grids, truth

    def _run_one(self, rep: int, grids) -> ReplicateResult:
        import torch
        seed = self.base_seed + rep
        sim = self.dgp.simulate(self.n_obs, seed)
        tcfg = TrainConfig(**{**self.train_config.__dict__, "seed": seed})
        trainer = Trainer(self.model_config, tcfg, device=torch.device(self.device))
        proc = DataProcessor(self.model_config)
        prepared = proc.prepare(sim.frame, trainer.ensemble, self.dgp.target_col)
        trainer.fit(prepared.inputs, prepared.y)

        link = make_link(self.model_config.link)
        if self.se_source == "laplace":
            ext = EffectExtractor.with_laplace(
                trainer.ensemble, prepared, link, self.centering)
        else:
            ext = EffectExtractor(trainer.ensemble, link,
                                  IntervalUQ(self.se_source), self.centering)
        est = {}
        comps = {}
        for name, grid in grids.items():
            e = ext.extract(name, grid, alpha=self.alpha)
            est[name] = {"mean": e.mean, "lo": e.lo, "hi": e.hi}
            c = getattr(ext, "last_laplace_components", None)
            if c is not None:
                comps[name] = c
        return ReplicateResult(seed=seed, estimates=est,
                               laplace_components=(comps or None),
                               fit_summary=trainer.fit_summary)

    def run(self, parallel: Optional[int] = None,
            progress: bool = True) -> StudyResult:
        grids, truth = self._grids_and_truth()
        result = StudyResult(truth=truth, grids=grids)

        def _maybe_progress(it):
            if progress:
                try:
                    from tqdm import tqdm
                    return tqdm(it, total=self.n_reps)
                except ImportError:
                    pass
            return it

        if parallel and parallel > 1:
            try:
                from joblib import Parallel, delayed
                reps = Parallel(n_jobs=parallel)(
                    delayed(self._run_one)(r, grids) for r in range(self.n_reps))
                result.replicates = list(reps)
                return result
            except ImportError:
                parallel = None

        for r in _maybe_progress(range(self.n_reps)):
            result.replicates.append(self._run_one(r, grids))
        return result

"""
evaluate.py — predictive performance metrics for a trained ensemble.

General over distribution (poisson | bernoulli) via the model's link. Computes
goodness-of-fit and predictive-interval coverage on the OUTCOME scale.

Note on the two kinds of coverage: what this reports is *predictive* coverage —
does the observed y fall inside a predictive band for the count/probability.
That is distinct from *estimator* coverage (does a CI for an effect curve contain
the true curve), which is what the Monte-Carlo study (Phase 6) measures. They
answer different questions and are deliberately kept separate.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             roc_auc_score)


def predict_ensemble(ensemble, inputs, chunk=4096):
    """Per-member predictions (M, N) and their mean (N,). Chunked to bound
    forward-pass activation memory on large N."""
    device = next(ensemble[0].parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    N = next(iter(inputs.values())).shape[0]
    preds = []
    with torch.no_grad():
        for m in ensemble:
            m.eval()
            parts = []
            for s in range(0, N, chunk):
                sl = slice(s, min(s + chunk, N))
                bi = {k: v[sl] for k, v in inputs.items()}
                parts.append(m(bi).cpu())
            preds.append(torch.cat(parts).numpy().ravel())
    preds = np.asarray(preds)
    return preds, preds.mean(axis=0)


class PerformanceEvaluator:
    def __init__(self, ensemble, distribution="poisson", alpha=0.05):
        self.ensemble = ensemble
        self.distribution = distribution
        self.alpha = alpha

    def evaluate(self, inputs, y) -> dict:
        y_true = (y if isinstance(y, np.ndarray)
                  else y.detach().cpu().numpy()).ravel()
        _, mu = predict_ensemble(self.ensemble, inputs)
        muc = np.clip(mu, 1e-8, None)
        z = stats.norm.ppf(1 - self.alpha / 2)
        dist = self.distribution

        # deviance + null deviance + McFadden R^2
        if dist == "poisson":
            ylog = np.where(y_true > 0, y_true * np.log(y_true / muc), 0.0)
            dev = 2 * np.sum(ylog - (y_true - muc))
            mu0 = np.clip(np.full_like(y_true, y_true.mean()), 1e-8, None)
            ylog0 = np.where(y_true > 0, y_true * np.log(y_true / mu0), 0.0)
            null_dev = 2 * np.sum(ylog0 - (y_true - mu0))
            phi = float(np.mean((y_true - mu) ** 2 / muc))
        else:
            mb = np.clip(mu, 1e-7, 1 - 1e-7)
            dev = 2 * np.sum(-y_true * np.log(mb) - (1 - y_true) * np.log(1 - mb))
            p0 = np.clip(y_true.mean(), 1e-7, 1 - 1e-7)
            null_dev = 2 * np.sum(-y_true * np.log(p0) - (1 - y_true) * np.log(1 - p0))
            phi = 1.0

        out = {
            "distribution": dist,
            "RMSE": float(np.sqrt(mean_squared_error(y_true, mu))),
            "MAE": float(mean_absolute_error(y_true, mu)),
            "Null_RMSE": float(np.sqrt(mean_squared_error(
                y_true, np.full_like(y_true, y_true.mean())))),
            "Deviance": float(dev),
            "Null_Deviance": float(null_dev),
            "McFadden_R2": float(1 - dev / null_dev),
            "Phi": phi,
            "alpha": self.alpha,
            "Nominal_Coverage": 1 - self.alpha,
        }

        # predictive Wald interval on the outcome scale
        if dist == "poisson":
            se = np.sqrt(phi * muc)
        else:
            se = np.sqrt(np.clip(mu * (1 - mu), 1e-8, None))
        lo, hi = mu - z * se, mu + z * se
        out["Predictive_Coverage"] = float(np.mean((y_true >= lo) & (y_true <= hi)))

        if dist == "bernoulli":
            out["Brier"] = float(np.mean((y_true - mu) ** 2))
            try:
                out["AUC"] = float(roc_auc_score(y_true, mu))
            except ValueError:
                out["AUC"] = float("nan")
        return out

    def report(self, m: dict) -> None:
        sep = "=" * 56
        nom = int(round(m["Nominal_Coverage"] * 100))
        print(f"\n{sep}\n  PERFORMANCE  [{m['distribution'].upper()}]\n{sep}")
        print(f"  RMSE         {m['RMSE']:.4f}   (null {m['Null_RMSE']:.4f})")
        print(f"  MAE          {m['MAE']:.4f}")
        print(f"  Deviance     {m['Deviance']:.2f}  (null {m['Null_Deviance']:.2f})")
        print(f"  McFadden R2  {m['McFadden_R2']:.4f}")
        if m["distribution"] == "poisson":
            print(f"  Phi          {m['Phi']:.4f}")
        else:
            print(f"  AUC          {m.get('AUC', float('nan')):.4f}")
            print(f"  Brier        {m.get('Brier', float('nan')):.4f}")
        print(f"  {nom}% predictive coverage: {m['Predictive_Coverage']*100:.2f}%")
        print(f"{sep}\n")

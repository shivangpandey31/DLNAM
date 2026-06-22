"""
visualize.py — plots driven by EffectExtractor / EffectEstimate.

General over term type and distribution. The visualizer never recomputes
centering or interval math (that is EffectExtractor's job) and never reaches into
model internals except through public term methods. A ground-truth EffectEstimate
can be overlaid on any effect plot, which is what the simulation studies use.

Dispatch by term type:
  surface  -> cumulative RR curve (+ optional lag-surface contour)
  smooth   -> 1-D RR curve
  trend    -> 1-D RR curve over normalised time
  categorical -> per-level RR segments
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .inference import EffectExtractor, EffectEstimate
from .terms.surface import SurfaceTerm
from .terms.smooth import TrendTerm
from .terms.categorical import CategoricalTerm


class ResultVisualizer:
    def __init__(self, ensemble, link, uq, centering, distribution="poisson",
                 labels=None, trainer=None):
        self.ensemble = ensemble
        self.link = link
        self.centering = centering
        self.distribution = distribution
        self.labels = labels or {}
        self.trainer = trainer
        self.ext = EffectExtractor(ensemble, link, uq, centering)

    def _label(self, name):
        return self.labels.get(name, name)

    def _yl(self):
        return "Relative Risk" if self.distribution == "poisson" else "Odds Ratio"

    # Distinct colour per term role so covariates don't read as the exposure RR.
    TYPE_COLORS = {
        "surface":     "#B22222",   # exposure — firebrick
        "smooth":      "#2C6E8F",   # confounders — steel blue
        "trend":       "#1B7837",   # trend — green
        "categorical": "#4A4A6A",   # categories — slate
    }

    def _color(self, name):
        term = self.ensemble[0].term(name)
        if isinstance(term, SurfaceTerm):
            return self.TYPE_COLORS["surface"]
        if isinstance(term, TrendTerm):
            return self.TYPE_COLORS["trend"]
        if isinstance(term, CategoricalTerm):
            return self.TYPE_COLORS["categorical"]
        return self.TYPE_COLORS["smooth"]

    # --- public API ------------------------------------------------------
    def plot_all(self, grids=None, truths=None):
        """One figure per term, dispatched by type, plus the training loss.
        grids/truths are optional dicts keyed by term name."""
        grids = grids or {}
        truths = truths or {}
        for name, term in self.ensemble[0].terms.items():
            g = grids.get(name)
            t = truths.get(name)
            if isinstance(term, CategoricalTerm):
                self.plot_categorical(name, truth=t)
            else:
                self.plot_effect(name, grid_raw=g, truth=t)
                if isinstance(term, SurfaceTerm):
                    self.plot_surface(name, grid_raw=g)
        if self.trainer is not None:
            self.plot_training_loss()

    def plot_effect(self, name, grid_raw=None, truth: Optional[EffectEstimate] = None,
                    ax=None, color=None):
        est = self.ext.extract(name, grid_raw)
        if color is None:
            color = self._color(name)
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 5))
        ax.plot(est.grid_raw, est.mean, color=color, lw=1.6, label="Ensemble mean")
        if est.lo is not None:
            ax.fill_between(est.grid_raw, est.lo, est.hi, color=color, alpha=0.18,
                            label=est.ci_label or "CI")
        if truth is not None:
            ax.plot(truth.grid_raw, truth.mean, "k--", lw=2, label="Truth")
        ax.axhline(1, color="0.4", lw=0.8)
        ax.set_xlabel(self._label(name)); ax.set_ylabel(self._yl())
        ax.set_title(f"{self._yl()}: {self._label(name)}")
        ax.legend(frameon=True); ax.grid(alpha=0.2); ax.margins(x=0)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_categorical(self, name, truth: Optional[EffectEstimate] = None, ax=None):
        est = self.ext.extract(name)
        term = self.ensemble[0].term(name)
        labels = term.order
        x = np.arange(len(est.mean)); W = 0.2
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(max(8, len(x)), 5))
        for i in x:
            if est.lo is not None:
                ax.fill_between([i - W, i + W], est.lo[i], est.hi[i],
                                color="darkslategray", alpha=0.15)
            ax.plot([i - W, i + W], [est.mean[i]] * 2, color="darkslategray", lw=1.6)
        if truth is not None:
            ax.plot(x, truth.mean, "kx", ms=8, label="Truth")
            ax.legend(frameon=True)
        ax.axhline(1, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(self._yl()); ax.set_title(f"{self._yl()}: {self._label(name)}")
        ax.grid(alpha=0.2)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_surface(self, name, grid_raw=None, ax=None):
        term = self.ensemble[0].term(name)
        if grid_raw is None:
            grid_raw = term.default_grid()
        ref = self.centering.value if self.centering.value is not None \
            else term._data_median
        # ensemble-mean per-lag log-RR -> RR
        stack = []
        for m in self.ensemble:
            stack.append(m.term(name).per_lag_log_rr(grid_raw, ref))
        rr = np.exp(np.mean(stack, axis=0))                  # (n_lags, G)
        lags = np.arange(rr.shape[0])
        V, Lg = np.meshgrid(grid_raw, lags)
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 5))
        c = ax.contourf(V, Lg, rr, levels=60, cmap="RdBu_r")
        plt.colorbar(c, ax=ax, label=self._yl())
        ax.set_xlabel(self._label(name)); ax.set_ylabel("Lag")
        ax.set_title(f"{self._yl()} surface: {self._label(name)}")
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_surface_3d(self, name, grid_raw=None, ax=None):
        """3-D RR surface over (value, lag). Same data as plot_surface, lifted
        as a height surface. Colour centred at RR = 1."""
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        import matplotlib.cm as cm
        term = self.ensemble[0].term(name)
        if grid_raw is None:
            grid_raw = term.default_grid()
        ref = self.centering.value if self.centering.value is not None \
            else term._data_median
        rr = np.exp(np.mean(
            [m.term(name).per_lag_log_rr(grid_raw, ref) for m in self.ensemble],
            axis=0))                                          # (n_lags, G)
        lags = np.arange(rr.shape[0])
        V, Lg = np.meshgrid(grid_raw, lags)

        vmin, vmax = min(rr.min(), 0.999), max(rr.max(), 1.001)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            "dlnam", ["#0a1628", "#ffffff", "#B22222"], N=256)

        standalone = ax is None
        if standalone:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        ax.plot_surface(V, Lg, rr, facecolors=cmap(norm(rr)), edgecolor="none",
                        shade=False, rcount=120, ccount=120, antialiased=False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label=self._yl())
        ax.set_xlabel(self._label(name)); ax.set_ylabel("Lag"); ax.set_zlabel(self._yl())
        ax.set_title(f"{self._yl()} surface: {self._label(name)}")
        ax.view_init(elev=18, azim=-60)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_training_loss(self, ax=None):
        if not getattr(self.trainer, "loss_history", None):
            print("no loss history"); return
        hist = self.trainer.loss_history
        epochs = [e for e, _ in hist[0]]
        losses = np.array([[l for _, l in h] for h in hist])
        w = max(1, int(0.05 * len(epochs)))
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 5))
        mean = losses.mean(0)[w:]; sd = losses.std(0)[w:]; ep = epochs[w:]
        ax.plot(ep, mean, color="#FF8C00", lw=1.5, label="Ensemble mean")
        ax.fill_between(ep, mean - sd, mean + sd, color="#FFAD60", alpha=0.35,
                        label="Ensemble spread")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training loss")
        ax.legend(frameon=True); ax.grid(alpha=0.2)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

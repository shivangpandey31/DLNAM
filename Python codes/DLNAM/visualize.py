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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .inference import EffectExtractor, EffectEstimate
from .terms.surface import SurfaceTerm
from .terms.smooth import TrendTerm
from .terms.categorical import CategoricalTerm


SURFACE_CMAP_COLOURS = ["#17151C", "#33283B", "#403249", "#FFFFFF",
                        "#C08A7C", "#A04F62", "#7A1832"]
SURFACE_CMAP_STOPS = [0.00, 0.48, 0.49, 0.50, 0.51, 0.52, 1.00]


class _AdaptiveRRNorm(mcolors.Normalize):
    def __init__(self, anchors_rr, anchors_pos=None, clip=True):
        self.anchors_rr = np.asarray(anchors_rr, dtype=float)
        self.anchors_pos = np.asarray(
            SURFACE_CMAP_STOPS if anchors_pos is None else anchors_pos, dtype=float
        )
        super().__init__(
            vmin=float(self.anchors_rr[0]),
            vmax=float(self.anchors_rr[-1]),
            clip=clip,
        )

    def __call__(self, value, clip=None):
        masked = np.ma.asarray(value)
        data = masked.filled(np.nan)
        out = np.interp(data, self.anchors_rr, self.anchors_pos)
        out = np.clip(out, 0.0, 1.0)
        return np.ma.array(out, mask=np.ma.getmask(masked))

    def inverse(self, value):
        return np.interp(value, self.anchors_pos, self.anchors_rr)


def _surface_norm(values, center=1.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("surface plot requires finite RR values")
    data_min = float(np.nanmin(arr))
    data_max = float(np.nanmax(arr))
    below = arr[arr < center]
    above = arr[arr > center]
    span = max(data_max - data_min, abs(center) * 1e-3, 1e-6)
    eps = max(span * 1e-6, 1e-12)
    lo_end = min(data_min, center - eps)
    hi_end = max(data_max, center + eps)
    lo_span = max(center - lo_end, eps)
    hi_span = max(hi_end - center, eps)
    n_inner = (len(SURFACE_CMAP_STOPS) - 3) // 2
    if n_inner == 1:
        lo_probs = [0.90]
        hi_probs = [0.10]
        lo_default = [center - 0.10 * lo_span]
        hi_default = [center + 0.10 * hi_span]
    elif n_inner == 2:
        lo_probs = [0.50, 0.90]
        hi_probs = [0.10, 0.50]
        lo_default = [center - 0.50 * lo_span, center - 0.10 * lo_span]
        hi_default = [center + 0.10 * hi_span, center + 0.50 * hi_span]
    else:
        raise ValueError("surface colour stops must have 5 or 7 entries")

    lo_mid = (np.quantile(below, lo_probs) if below.size >= 10
              else np.asarray(lo_default, dtype=float))
    hi_mid = (np.quantile(above, hi_probs) if above.size >= 10
              else np.asarray(hi_default, dtype=float))
    lo_mid = np.clip(lo_mid, lo_end + eps, center - eps)
    hi_mid = np.clip(hi_mid, center + eps, hi_end - eps)
    anchors = np.r_[lo_end, lo_mid, center, hi_mid, hi_end]
    for i in range(1, anchors.size):
        if anchors[i] <= anchors[i - 1]:
            anchors[i] = anchors[i - 1] + eps
    return _AdaptiveRRNorm(anchors)


def _surface_colorbar_ticks(norm):
    if isinstance(norm, _AdaptiveRRNorm):
        ticks = np.array([norm.anchors_rr[0], 1.0, norm.anchors_rr[-1]])
        return ticks, [f"{x:.2f}" for x in ticks]
    return None, None


def _surface_cmap(name="dlnam_rr_surface", colours=None, stops=None, n=2048):
    colours = SURFACE_CMAP_COLOURS if colours is None else colours
    stops = SURFACE_CMAP_STOPS if stops is None else stops
    xs = np.linspace(0.0, 1.0, n)
    pos = np.asarray(stops, dtype=float)
    rgb = np.asarray([mcolors.to_rgb(c) for c in colours], dtype=float)
    out = np.empty((n, 3), dtype=float)
    for i in range(len(pos) - 1):
        lo, hi = pos[i], pos[i + 1]
        mask = (xs >= lo) & (xs <= hi) if i == len(pos) - 2 else (xs >= lo) & (xs < hi)
        t = np.clip((xs[mask] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        out[mask] = (1.0 - t[:, None]) * rgb[i] + t[:, None] * rgb[i + 1]
    out[xs < pos[0]] = rgb[0]
    out[xs > pos[-1]] = rgb[-1]
    return mcolors.ListedColormap(out, name=name)

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
        "surface":     "#17151C",   # exposure surface
        "smooth":      "#7A1832",   # smooth covariates
        "trend":       "#A04F62",   # trend
        "categorical": "#C08A7C",   # categories
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
        norm = _surface_norm(rr)
        c = ax.contourf(V, Lg, rr, levels=60, cmap=_surface_cmap(), norm=norm)
        cb = plt.colorbar(c, ax=ax, label=self._yl())
        ticks, ticklabels = _surface_colorbar_ticks(norm)
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklabels)
        ax.set_xlabel(self._label(name)); ax.set_ylabel("Lag")
        ax.set_title(f"{self._yl()} surface: {self._label(name)}")
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_surface_3d(self, name, grid_raw=None, ax=None):
        """3-D RR surface over (value, lag). Same data as plot_surface, lifted
        as a height surface. Colour uses the same adaptive RR scale."""
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

        norm = _surface_norm(rr)
        cmap = _surface_cmap("dlnam_rr_surface")

        standalone = ax is None
        if standalone:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        ax.plot_surface(V, Lg, rr, facecolors=cmap(norm(rr)), edgecolor="none",
                        shade=False, rcount=120, ccount=120, antialiased=False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label=self._yl())
        ticks, ticklabels = _surface_colorbar_ticks(norm)
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklabels)
        ax.set_xlabel(self._label(name)); ax.set_ylabel("Lag"); ax.set_zlabel(self._yl())
        ax.set_title(f"{self._yl()} surface: {self._label(name)}")
        ax.view_init(elev=18, azim=-60)
        if standalone:
            plt.tight_layout(); plt.show()
        return ax

    def plot_training_loss(self, ax=None):
        if not getattr(self.trainer, "loss_history", None):
            print("No loss history to plot.")
            return
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

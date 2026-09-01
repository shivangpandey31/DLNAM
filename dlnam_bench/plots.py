"""
dlnam_bench.plots -- publication figures for the DLNAM paper experiments.

Single 3x2 grid (save_all -> mc_grid.{pdf,png}):
    row 0  Error     : RMSE per model, partitioned by exact MSE-share into
                       bias^2 (solid) and variance (light hatch); marker with
                       95% Monte Carlo CI
                       at the RMSE tip. Columns = Total / Interior / Boundary.
    row 1  Coverage  : pointwise coverage vs nominal 0.95. Same columns.

Exact decomposition: MSE = bias^2 + variance and RMSE = sqrt(MSE). From the
reported bias^2 and variance, the bar total is RMSE and the split is the exact
fraction of MSE from each component.

Consumes the runner's results dict:
    results[scenario][model]["{metric}_{region}"]      value
    results[scenario][model]["{metric}_{region}_se"]   Monte-Carlo SE
    model  in {"DLNAM","QAIC","QBIC","Penalised","TDLNM"} as available
    region in {"tot","int","bnd"}
    metric "err" (RMSE, logRR), "bias2", "var" (logRR), "cov" (+ _se as stored).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

# --- visual system (muted burgundy rose) ------------------------------------
COLOURS = {
    "DLNAM":     "#17151C",  # near-black ink/plum
    "QAIC":      "#403249",  # muted plum
    "QBIC":      "#7A1832",  # deep burgundy
    "Penalised": "#A04F62",  # muted rose-burgundy
    "TDLNM":     "#C08A7C",  # desaturated rose
}
TRUTH_COLOUR = "#000000"       # dense dotted DGP line
TRUTH_LINESTYLE = (0, (0.45, 1.0))
TRUTH_LW = 1.35
TRUTH_PATH_EFFECTS = []
SURFACE_COLOURS = [
    "#17151C", "#17151C", "#241F2A", "#33283B", "#403249",
    "#FFFFFF",
    "#C08A7C", "#A04F62", "#8E2944", "#7A1832", "#7A1832",
]
SURFACE_STOPS = [0.00, 0.05, 0.20, 0.35, 0.47, 0.50,
                 0.53, 0.65, 0.80, 0.95, 1.00]
MARKERS = {"DLNAM": "o", "QAIC": "^", "QBIC": "s", "Penalised": "D", "TDLNM": "P"}
MODELS = ["DLNAM", "QAIC", "QBIC", "Penalised", "TDLNM"]
# Legend labels keep the DLNM family visible while remaining compact.
LABELS = {"DLNAM": "DLNAM", "QAIC": "DLNM (QAIC)",
          "QBIC": "DLNM (QBIC)", "Penalised": "P-DLNM",
          "TDLNM": "T-DLNM"}
REGIONS = [("Total", "tot"), ("Interior", "int"), ("Boundary", "bnd")]
NAMES = {
    "dgp1": "DGP 1",
    "dgp2": "DGP 2",
    "dgp3": "DGP 3",
    "dgp4": "DGP 4",
    "smooth": "DGP 1",
    "delayed_peaks": "DGP 2",
    "localized_peak": "DGP 3",
    "tilting_threshold": "DGP 4",
}

_RC = {
    "figure.dpi": 150, "savefig.dpi": 400,
    "font.size": 8.5, "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "hatch.linewidth": 0.5,
    "lines.dash_capstyle": "round", "lines.solid_capstyle": "round",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": False,
}


VARIANCE_ALPHA = 0.28   # variance-segment opacity, composited on white (lighter, same hue)
MC_SE_MULT = 1.96       # figures show approximate 95% Monte Carlo CIs.


class AdaptiveRRNorm(mpl.colors.Normalize):
    """Piecewise RR normalisation with RR=1 fixed at the neutral colour.

    Anchors are chosen from pooled values for a full figure, so the same colour
    has the same RR interpretation across panels while remaining readable when
    surfaces are tightly concentrated around RR=1.
    """

    def __init__(self, anchors_rr, anchors_pos=None, clip=True):
        self.anchors_rr = np.asarray(anchors_rr, dtype=float)
        self.anchors_pos = np.asarray(
            SURFACE_STOPS if anchors_pos is None else anchors_pos, dtype=float
        )
        if self.anchors_rr.shape != self.anchors_pos.shape:
            raise ValueError("anchors_rr and anchors_pos must have equal length")
        if np.any(np.diff(self.anchors_rr) <= 0):
            raise ValueError("anchors_rr must be strictly increasing")
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


def surface_norm(values, center=1.0, stops=None):
    """Adaptive figure-level RR normalisation with RR=1 mapped to white.

    The endpoints are the pooled finite min/max, so the colourbar spans the full
    realised RR range. Interior anchors are quantile-based to keep structure near
    RR=1 readable when most surface values are tightly concentrated.
    """
    stops = np.asarray(SURFACE_STOPS if stops is None else stops, dtype=float)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("surface_norm requires at least one finite RR value")

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

    n_inner = (len(stops) - 3) // 2
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
    elif n_inner == 4:
        lo_probs = [0.01, 0.15, 0.45, 0.75]
        hi_probs = [0.25, 0.60, 0.85, 0.99]
        lo_default = [center - 0.90 * lo_span, center - 0.60 * lo_span,
                      center - 0.30 * lo_span, center - 0.10 * lo_span]
        hi_default = [center + 0.10 * hi_span, center + 0.30 * hi_span,
                      center + 0.60 * hi_span, center + 0.90 * hi_span]
    else:
        raise ValueError("surface colour stops must have 5, 7, or 11 entries")

    if below.size >= 10:
        lo_mid = np.quantile(below, lo_probs)
    else:
        lo_mid = np.asarray(lo_default, dtype=float)
    if above.size >= 10:
        hi_mid = np.quantile(above, hi_probs)
    else:
        hi_mid = np.asarray(hi_default, dtype=float)

    lo_mid = np.clip(lo_mid, lo_end + eps, center - eps)
    hi_mid = np.clip(hi_mid, center + eps, hi_end - eps)
    anchors = np.r_[lo_end, lo_mid, center, hi_mid, hi_end]
    for i in range(1, anchors.size):
        if anchors[i] <= anchors[i - 1]:
            anchors[i] = anchors[i - 1] + eps
    return AdaptiveRRNorm(anchors, stops)


def surface_colorbar_ticks(norm):
    if isinstance(norm, AdaptiveRRNorm):
        return np.array([norm.anchors_rr[0], 1.0, norm.anchors_rr[-1]])
    return None


def format_rr_ticks(ticks):
    ticks = np.asarray(ticks, dtype=float)
    return [f"{x:.2f}" for x in ticks]


def surface_cmap(name="rr_surface", colours=None, stops=None, n=2048):
    """Smooth RR colormap with white at the neutral stop."""
    colours = SURFACE_COLOURS if colours is None else colours
    stops = SURFACE_STOPS if stops is None else stops
    xs = np.linspace(0.0, 1.0, n)
    pos = np.asarray(stops, dtype=float)
    rgb = np.asarray([mpl.colors.to_rgb(c) for c in colours], dtype=float)
    out = np.empty((n, 3), dtype=float)
    for i in range(len(pos) - 1):
        lo, hi = pos[i], pos[i + 1]
        mask = (xs >= lo) & (xs <= hi) if i == len(pos) - 2 else (xs >= lo) & (xs < hi)
        t = np.clip((xs[mask] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        out[mask] = (1.0 - t[:, None]) * rgb[i] + t[:, None] * rgb[i + 1]
    out[xs < pos[0]] = rgb[0]
    out[xs > pos[-1]] = rgb[-1]
    return mpl.colors.ListedColormap(out, name=name)


def _composite(hexc, alpha):
    """Alpha-composite the colour on white and return an OPAQUE rgb. Preserves hue
    (a lighter version of the same colour) while staying opaque, so gridlines and
    the axis frame do not bleed through the bars."""
    h = hexc.lstrip("#"); r, g, b = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return tuple(alpha * x + (1 - alpha) * 1.0 for x in (r, g, b))


class _WhiskerHandler(HandlerBase):
    """Legend entry as a capped horizontal whisker |--|."""
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        yc = yd + h / 2.0; c = "0.4"; lw = 1.1; cap = h * 0.38
        return [Line2D([xd, xd + w], [yc, yc], color=c, lw=lw, transform=trans),
                Line2D([xd, xd], [yc - cap, yc + cap], color=c, lw=lw, transform=trans),
                Line2D([xd + w, xd + w], [yc - cap, yc + cap], color=c, lw=lw, transform=trans)]


def _nm(s):
    return NAMES.get(s, s.replace("_", " ").title())


def _error_segments(row, region):
    """Return RMSE-scale bar segments for the stored MSE decomposition."""
    rmse = float(row[f"err_{region}"])
    var = float(row[f"var_{region}"])
    bias2 = row.get(f"bias2_{region}")
    if bias2 is None:
        bias2 = max(rmse ** 2 - var, 0.0)
    else:
        bias2 = max(float(bias2), 0.0)
    var = max(var, 0.0)
    mse = bias2 + var
    if rmse <= 0.0 or mse <= 0.0:
        return 0.0, 0.0
    return rmse * bias2 / mse, rmse * var / mse


def order_by_difficulty(results, region="int", ref=None):
    scen = list(results.keys())
    present = [m for m in MODELS if m in next(iter(results.values()))]
    if ref is None or ref not in present:
        ref = present[0]          # hero = first model on the axis (generic)
    others = [m for m in present if m != ref]
    def gap(s):
        r = results[s][ref][f"err_{region}"]
        worst = max(results[s][m][f"err_{region}"] for m in others) if others else r
        return worst / max(r, 1e-12)
    return sorted(scen, key=gap)


def mc_grid(results, scenarios=None, title="Simulation Study: Model Comparison"):
    scenarios = scenarios or list(results.keys())
    present = [m for m in MODELS if m in next(iter(results.values()))]
    n_s = len(scenarios)
    y0 = np.arange(n_s)[::-1]
    off = np.linspace(0.26, -0.26, len(present))
    bh = 0.13

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharey=True)

        # shared RMSE x-limit (sqrt compresses range enough for cross-region compare)
        xmax = max(results[s][m][f"err_{rt}"] + MC_SE_MULT * results[s][m].get(f"err_{rt}_se", 0.0)
                   for s in scenarios for m in present for _, rt in REGIONS) * 1.06

        # ---- row 0: error (RMSE partitioned by exact MSE-share) ----
        for (rname, rt), ax in zip(REGIONS, axes[0]):
            for mi, m in enumerate(present):
                for si, s in enumerate(scenarios):
                    y = y0[si] + off[mi]
                    rmse = results[s][m][f"err_{rt}"]
                    ese = results[s][m].get(f"err_{rt}_se", 0.0)
                    bias_seg, var_seg = _error_segments(results[s][m], rt)
                    ax.barh(y, bias_seg, height=bh, color=COLOURS[m],
                            edgecolor="none", lw=0, zorder=3)
                    ax.barh(y, var_seg, left=bias_seg, height=bh,
                            facecolor=_composite(COLOURS[m], VARIANCE_ALPHA),
                            edgecolor=COLOURS[m],
                            lw=0, hatch="//", zorder=3)
                    ax.errorbar(rmse, y, xerr=MC_SE_MULT * ese, fmt=MARKERS[m], ms=4.2,
                                mfc=COLOURS[m], mec=COLOURS[m],
                                ecolor=COLOURS[m], elinewidth=0.9,
                                capsize=1.8, zorder=5, clip_on=True)
            ax.set_xlim(0, xmax); ax.set_title(rname, fontsize=10, weight="bold", pad=8)
            ax.grid(axis="x", color="0.92", lw=0.5)
            ax.tick_params(length=0, axis="y")
            ax.set_xlabel("RMSE", fontsize=8.5)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))

        # ---- row 1: coverage vs nominal 0.95 ----
        # Coverage is a proportion, so keep the axis anchored at 0.
        for (rname, rt), ax in zip(REGIONS, axes[1]):
            for mi, m in enumerate(present):
                xs = [results[s][m][f"cov_{rt}"] for s in scenarios]
                xe = [MC_SE_MULT * results[s][m].get(f"cov_{rt}_se", 0.0) for s in scenarios]
                ys = [y0[si] + off[mi] for si in range(n_s)]
                ax.errorbar(xs, ys, xerr=xe, fmt=MARKERS[m], ms=4.2, mfc=COLOURS[m],
                            mec=COLOURS[m], ecolor=COLOURS[m], elinewidth=0.9,
                            capsize=1.8, lw=0, zorder=3, clip_on=False)
            ax.set_xlim(0.0, 1.02)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 0.95])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "0.95"])
            ax.grid(axis="x", color="0.92", lw=0.5)
            ax.tick_params(length=0, axis="y")
            ax.set_xlabel("Pointwise Coverage", fontsize=8.5)

        for ax in axes.flat:
            for sp in ax.spines.values():
                sp.set_zorder(10)          # frame drawn above bars/whiskers

        for ax in (axes[0][0], axes[1][0]):
            ax.set_yticks(y0); ax.set_yticklabels([_nm(s) for s in scenarios])
        axes[0][0].annotate("Error", (-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center", fontsize=10, weight="bold")
        axes[1][0].annotate("Coverage", (-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center", fontsize=10, weight="bold")

        model_h = [Line2D([0], [0], marker=MARKERS[m], color=COLOURS[m], lw=0, ms=6,
                          label=LABELS[m]) for m in present]
        se_proxy = Line2D([0], [0], color="0.4", lw=1.1, label="95% MC CI")
        comp_h = [Patch(facecolor="0.35", edgecolor="none", label="Bias\u00b2 (%)"),
                  Patch(facecolor=_composite("#666666", VARIANCE_ALPHA + 0.25),
                        edgecolor="0.35", lw=0,
                        hatch="//", label="Variance (%)"), se_proxy]
        legend_h = model_h + comp_h
        fig.legend(handles=legend_h, loc="lower center", ncol=len(legend_h),
                   bbox_to_anchor=(0.5, -0.06), fontsize=7.8,
                   columnspacing=1.05, handletextpad=0.45,
                   handler_map={se_proxy: _WhiskerHandler()})
        fig.suptitle(title, fontsize=11, y=1.0, weight="bold")
        fig.tight_layout(rect=(0.02, 0.04, 1, 0.98))
    return fig


def save_all(results, outdir, scenarios=None, fmts=("pdf", "png"),
             curves=None, boundary=None, stem=None,
             title="Simulation Study: Model Comparison"):
    """Write the main Monte-Carlo figure.

    If `curves` is provided (per-scenario MC-mean cumulative-RR arrays), the merged
    A/B/C composite is written; otherwise the standalone 3x2 grid. `curves` schema:
        {scenario: {"grid","truth", <model>: 1-D array, ...}}
    `boundary` (optional): {scenario: (q_lo, q_hi)} exposure-boundary marks for the
    A-row gridlines; defaults to inner 12%/88% of the grid if omitted.
    `stem` (optional): output filename stem (no extension). Defaults to
    "mc_composite" (composite) or "mc_grid" (grid) -- callers running variant
    studies pass e.g. stem="exu_eval" / "ablation" to name outputs directly.
    """
    scenarios = scenarios or order_by_difficulty(results)
    if curves is not None:
        fig = composite(results, curves, scenarios, boundary, title=title)
        stem = stem or "mc_composite"
    else:
        fig = mc_grid(results, scenarios, title=title)
        stem = stem or "mc_grid"
    paths = []
    for ext in fmts:
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=400)
        paths.append(p)
    plt.close(fig)
    return paths


# ===========================================================================
# Merged A/B/C composite: A curves (1x4) + B error (1x3) + C coverage (1x3).
# A shows truth + MC-mean curve per method with faint boundary-region shading;
# B and C reproduce the mc_grid error/coverage panels exactly. One title, one
# shared legend; metric-panel whiskers show estimate ± Monte Carlo SE.
# ===========================================================================

_A_LABEL_SZ = 8.5
_BOUNDARY_TINT = "#DCE6EC"      # lightest blue-grey (near white), palette hue
_BOUNDARY_ALPHA = 0.45


def _draw_error_block(axes_row, results, scenarios, present, y0, off, bh, xmax):
    for (rname, rt), ax in zip(REGIONS, axes_row):
        for mi, m in enumerate(present):
            for si, s in enumerate(scenarios):
                y = y0[si] + off[mi]
                rmse = results[s][m][f"err_{rt}"]
                ese = results[s][m].get(f"err_{rt}_se", 0.0)
                bias_seg, var_seg = _error_segments(results[s][m], rt)
                ax.barh(y, bias_seg, height=bh, color=COLOURS[m],
                        edgecolor="none", lw=0, zorder=3)
                ax.barh(y, var_seg, left=bias_seg, height=bh,
                        facecolor=_composite(COLOURS[m], VARIANCE_ALPHA),
                        edgecolor=COLOURS[m], lw=0, hatch="//", zorder=3)
                ax.errorbar(rmse, y, xerr=MC_SE_MULT * ese, fmt=MARKERS[m], ms=4.2,
                            mfc=COLOURS[m], mec=COLOURS[m], ecolor=COLOURS[m],
                            elinewidth=0.9, capsize=1.8, zorder=5, clip_on=True)
        ax.set_xlim(0, xmax)
        ax.set_title(rname, fontsize=9, weight="bold", loc="center", pad=6)
        ax.grid(axis="x", color="0.92", lw=0.5)
        ax.tick_params(length=0, axis="y")
        ax.set_xlabel("RMSE", fontsize=8.5)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        for sp in ax.spines.values():
            sp.set_zorder(10)


def _draw_coverage_block(axes_row, results, scenarios, present, y0, off):
    # Coverage is a proportion, so keep the axis anchored at 0.
    for (rname, rt), ax in zip(REGIONS, axes_row):
        for mi, m in enumerate(present):
            xs = [results[s][m][f"cov_{rt}"] for s in scenarios]
            xe = [MC_SE_MULT * results[s][m].get(f"cov_{rt}_se", 0.0) for s in scenarios]
            ys = [y0[si] + off[mi] for si in range(len(scenarios))]
            ax.errorbar(xs, ys, xerr=xe, fmt=MARKERS[m], ms=4.2, mfc=COLOURS[m],
                        mec=COLOURS[m], ecolor=COLOURS[m], elinewidth=0.9,
                        capsize=1.8, lw=0, zorder=3, clip_on=False)
        ax.set_xlim(0.0, 1.02)
        ax.set_title(rname, fontsize=9, weight="bold", loc="center", pad=6)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 0.95])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "0.95"])
        ax.grid(axis="x", color="0.92", lw=0.5)
        ax.tick_params(length=0, axis="y")
        ax.set_xlabel("Pointwise Coverage", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_zorder(10)


def _draw_curve_block(axes_row, curves, scenarios, boundary):
    for s, ax in zip(scenarios, axes_row):
        c = curves[s]
        grid = np.asarray(c["grid"])
        if boundary and s in boundary:
            q_lo, q_hi = boundary[s]
        else:
            q_lo = grid[0] + 0.12 * (grid[-1] - grid[0])
            q_hi = grid[0] + 0.88 * (grid[-1] - grid[0])
        ax.axvline(q_lo, color="0.92", lw=0.5, zorder=0)
        ax.axvline(q_hi, color="0.92", lw=0.5, zorder=0)
        ax.axhline(1.0, color="0.92", lw=0.5, zorder=0)
        present = [m for m in MODELS if m in c]
        # Draw in legend order, with the DGP on top, then DLNAM, then the DLNM
        # family. The explicit z-order also keeps model curves below the frame.
        for i, m in enumerate(present):
            z = 2 + len(present) - i
            ax.plot(grid, c[m], color=COLOURS[m], lw=0.75, zorder=z)
        truth_line, = ax.plot(grid, c["truth"], color=TRUTH_COLOUR,
                              ls=TRUTH_LINESTYLE, lw=TRUTH_LW,
                              zorder=3 + len(present))
        truth_line.set_path_effects(TRUTH_PATH_EFFECTS)
        ax.set_title(_nm(s), fontsize=9, weight="bold", loc="center", pad=6)
        ax.set_xlabel("Exposure", fontsize=_A_LABEL_SZ)
        ax.margins(x=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_zorder(20)


def composite(results, curves, scenarios=None, boundary=None,
              title="Simulation Study: Model Comparison"):
    scenarios = scenarios or list(results.keys())
    present = [m for m in MODELS if m in next(iter(results.values()))]
    n_s = len(scenarios)
    y0 = np.arange(n_s)[::-1]
    off = np.linspace(0.26, -0.26, len(present))
    bh = 0.13

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.5, 9.4))
        gs = fig.add_gridspec(3, 12, height_ratios=[0.72, 1.15, 1.15],
                              hspace=0.52, wspace=0.75,
                              left=0.07, right=0.985, top=0.9, bottom=0.075)
        axA = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(4)]
        axB = [fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(3)]
        axC = [fig.add_subplot(gs[2, 4 * i:4 * i + 4]) for i in range(3)]
        for i in (1, 2):
            axB[i].sharey(axB[0]); axC[i].sharey(axC[0])

        _draw_curve_block(axA, curves, scenarios, boundary)
        axA[0].set_ylabel("Cumulative RR", fontsize=_A_LABEL_SZ)

        xmax = max(results[s][m][f"err_{rt}"] + MC_SE_MULT * results[s][m].get(f"err_{rt}_se", 0.0)
                   for s in scenarios for m in present for _, rt in REGIONS) * 1.06
        _draw_error_block(axB, results, scenarios, present, y0, off, bh, xmax)
        _draw_coverage_block(axC, results, scenarios, present, y0, off)
        for ax in (axB[0], axC[0]):
            ax.set_yticks(y0); ax.set_yticklabels([_nm(s) for s in scenarios])
        for ax in (axB[1], axB[2], axC[1], axC[2]):
            ax.tick_params(labelleft=False)

        fig.canvas.draw()
        for ax, letter, name in [(axA[0], "A", "Estimated Curves"),
                                 (axB[0], "B", "Error"),
                                 (axC[0], "C", "Coverage")]:
            top = ax.get_position().y1
            fig.text(0.012, top + 0.028, f"{letter}  {name}", fontsize=11,
                     weight="bold", va="bottom", ha="left")

        model_h = [Line2D([0], [0], marker=MARKERS[m], color=COLOURS[m], lw=0, ms=6,
                          label=LABELS[m]) for m in present]
        truth_h = Line2D([0], [0], color=TRUTH_COLOUR,
                         ls=TRUTH_LINESTYLE, lw=1.5, label="DGP")
        se_proxy = Line2D([0], [0], color="0.4", lw=1.1, label="95% MC CI")
        comp_h = [Patch(facecolor="0.35", edgecolor="none", label="Bias\u00b2 (%)"),
                  Patch(facecolor=_composite("#666666", VARIANCE_ALPHA + 0.25),
                        edgecolor="0.35", lw=0, hatch="//", label="Variance (%)"),
                  se_proxy]
        legend_h = [truth_h] + model_h + comp_h
        fig.legend(handles=legend_h, loc="lower center", ncol=len(legend_h),
                   bbox_to_anchor=(0.5, -0.025), fontsize=7.8, columnspacing=1.05,
                   handletextpad=0.5, handler_map={se_proxy: _WhiskerHandler()})
        fig.suptitle(title, fontsize=13, weight="bold", y=0.96)
    return fig


def metric_composite(
    results,
    scenarios=None,
    title="Simulation Study: Exposure-Lag Surface",
):
    """Render the error and coverage rows using the main composite style."""
    scenarios = scenarios or list(results.keys())
    present = [m for m in MODELS if m in next(iter(results.values()))]
    n_s = len(scenarios)
    y0 = np.arange(n_s)[::-1]
    off = np.linspace(0.26, -0.26, len(present))
    bh = 0.13

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.5, 6.7))
        gs = fig.add_gridspec(
            2,
            12,
            height_ratios=[1.15, 1.15],
            hspace=0.52,
            wspace=0.75,
            left=0.07,
            right=0.985,
            top=0.88,
            bottom=0.12,
        )
        axA = [fig.add_subplot(gs[0, 4 * i:4 * i + 4]) for i in range(3)]
        axB = [fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(3)]
        for i in (1, 2):
            axA[i].sharey(axA[0])
            axB[i].sharey(axB[0])

        xmax = max(
            results[s][m][f"err_{rt}"]
            + MC_SE_MULT * results[s][m].get(f"err_{rt}_se", 0.0)
            for s in scenarios
            for m in present
            for _, rt in REGIONS
        ) * 1.06
        _draw_error_block(axA, results, scenarios, present, y0, off, bh, xmax)
        _draw_coverage_block(axB, results, scenarios, present, y0, off)
        for ax in (axA[0], axB[0]):
            ax.set_yticks(y0)
            ax.set_yticklabels([_nm(s) for s in scenarios])
        for ax in (axA[1], axA[2], axB[1], axB[2]):
            ax.tick_params(labelleft=False)

        fig.canvas.draw()
        for ax, letter, name in [
            (axA[0], "A", "Error"),
            (axB[0], "B", "Coverage"),
        ]:
            top = ax.get_position().y1
            fig.text(
                0.012,
                top + 0.028,
                f"{letter}  {name}",
                fontsize=11,
                weight="bold",
                va="bottom",
                ha="left",
            )

        model_h = [
            Line2D(
                [0],
                [0],
                marker=MARKERS[m],
                color=COLOURS[m],
                lw=0,
                ms=6,
                label=LABELS[m],
            )
            for m in present
        ]
        se_proxy = Line2D([0], [0], color="0.4", lw=1.1, label="95% MC CI")
        comp_h = [
            Patch(facecolor="0.35", edgecolor="none", label="Bias\u00b2 (%)"),
            Patch(
                facecolor=_composite("#666666", VARIANCE_ALPHA + 0.25),
                edgecolor="0.35",
                lw=0,
                hatch="//",
                label="Variance (%)",
            ),
            se_proxy,
        ]
        legend_h = model_h + comp_h
        fig.legend(
            handles=legend_h,
            loc="lower center",
            ncol=len(legend_h),
            bbox_to_anchor=(0.5, -0.018),
            fontsize=7.8,
            columnspacing=1.05,
            handletextpad=0.5,
            handler_map={se_proxy: _WhiskerHandler()},
        )
        fig.suptitle(title, fontsize=13, weight="bold", y=0.96)
    return fig


def save_metric_composite(
    results,
    outdir,
    scenarios=None,
    fmts=("pdf", "png"),
    stem="mc_model_comparison_surface",
    title="Simulation Study: Exposure-Lag Surface",
):
    """Write a standalone error-and-coverage composite."""
    fig = metric_composite(results, scenarios=scenarios, title=title)
    paths = []
    for ext in fmts:
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=400)
        paths.append(path)
    plt.close(fig)
    return paths


# ===========================================================================
# Joint-fit composite: the A/B/C figure plus D = RMSE degradation relative to
# the corresponding single-exposure MC.
# ===========================================================================

def _draw_degradation_block(axes_row, degradation, scenarios, present, y0, off):
    # The axis must reach the end of the longest whisker, as the error block
    # does: ratios and standard errors are on different scales, so the limit is
    # taken over interval endpoints rather than over the two mixed together
    # (clip_on is False here, so an over-long interval would cross the spine).
    ends = []
    for s in scenarios:
        if s not in degradation:
            continue
        for m in present:
            if m not in degradation[s]:
                continue
            for _, rt in REGIONS:
                v = degradation[s][m].get(f"deg_{rt}", np.nan)
                e = degradation[s][m].get(f"deg_{rt}_se", 0.0)
                if np.isfinite(v):
                    ends.append(float(v) + MC_SE_MULT * float(e))
    xmax = max(1.15, max(ends) * 1.06) if ends else 1.15

    for (rname, rt), ax in zip(REGIONS, axes_row):
        for mi, m in enumerate(present):
            xs, xe, ys = [], [], []
            for si, s in enumerate(scenarios):
                if s not in degradation or m not in degradation[s]:
                    continue
                xs.append(degradation[s][m][f"deg_{rt}"])
                xe.append(MC_SE_MULT * degradation[s][m].get(f"deg_{rt}_se", 0.0))
                ys.append(y0[si] + off[mi])
            if xs:
                ax.errorbar(xs, ys, xerr=xe, fmt=MARKERS[m], ms=4.2,
                            mfc=COLOURS[m], mec=COLOURS[m], ecolor=COLOURS[m],
                            elinewidth=0.9, capsize=1.8, lw=0, zorder=3,
                            clip_on=False)
        ax.set_xlim(0.0, xmax)
        ax.set_title(rname, fontsize=9, weight="bold", loc="center", pad=6)
        ax.grid(axis="x", color="0.92", lw=0.5)
        ax.tick_params(length=0, axis="y")
        ax.set_xlabel("RMSE Degradation", fontsize=8.5)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        for sp in ax.spines.values():
            sp.set_zorder(10)


def composite_with_degradation(results, curves, degradation, scenarios=None, boundary=None,
                               title="Simulation Study: Model Comparison (Joint)"):
    scenarios = scenarios or list(results.keys())
    present = [m for m in MODELS if m in next(iter(results.values()))]
    n_s = len(scenarios)
    y0 = np.arange(n_s)[::-1]
    off = np.linspace(0.26, -0.26, len(present))
    bh = 0.13

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.5, 11.9))
        gs = fig.add_gridspec(4, 12, height_ratios=[0.70, 1.05, 1.05, 1.05],
                              hspace=0.55, wspace=0.75,
                              left=0.07, right=0.985, top=0.91, bottom=0.065)
        axA = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(4)]
        axB = [fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(3)]
        axC = [fig.add_subplot(gs[2, 4 * i:4 * i + 4]) for i in range(3)]
        axD = [fig.add_subplot(gs[3, 4 * i:4 * i + 4]) for i in range(3)]
        for i in (1, 2):
            axB[i].sharey(axB[0]); axC[i].sharey(axC[0]); axD[i].sharey(axD[0])

        _draw_curve_block(axA, curves, scenarios, boundary)
        axA[0].set_ylabel("Cumulative RR", fontsize=_A_LABEL_SZ)

        xmax = max(results[s][m][f"err_{rt}"] + MC_SE_MULT * results[s][m].get(f"err_{rt}_se", 0.0)
                   for s in scenarios for m in present for _, rt in REGIONS) * 1.06
        _draw_error_block(axB, results, scenarios, present, y0, off, bh, xmax)
        _draw_coverage_block(axC, results, scenarios, present, y0, off)
        _draw_degradation_block(axD, degradation, scenarios, present, y0, off)

        for ax in (axB[0], axC[0], axD[0]):
            ax.set_yticks(y0); ax.set_yticklabels([_nm(s) for s in scenarios])
        for ax in (axB[1], axB[2], axC[1], axC[2], axD[1], axD[2]):
            ax.tick_params(labelleft=False)

        fig.canvas.draw()
        for ax, letter, name in [(axA[0], "A", "Estimated Curves"),
                                 (axB[0], "B", "Error"),
                                 (axC[0], "C", "Coverage"),
                                 (axD[0], "D", "Degradation")]:
            top = ax.get_position().y1
            fig.text(0.012, top + 0.024, f"{letter}  {name}", fontsize=11,
                     weight="bold", va="bottom", ha="left")

        model_h = [Line2D([0], [0], marker=MARKERS[m], color=COLOURS[m], lw=0, ms=6,
                          label=LABELS[m]) for m in present]
        truth_h = Line2D([0], [0], color=TRUTH_COLOUR,
                         ls=TRUTH_LINESTYLE, lw=1.5, label="DGP")
        se_proxy = Line2D([0], [0], color="0.4", lw=1.1, label="95% MC CI")
        comp_h = [Patch(facecolor="0.35", edgecolor="none", label="Bias\u00b2 (%)"),
                  Patch(facecolor=_composite("#666666", VARIANCE_ALPHA + 0.25),
                        edgecolor="0.35", lw=0, hatch="//", label="Variance (%)"),
                  se_proxy]
        legend_h = [truth_h] + model_h + comp_h
        fig.legend(handles=legend_h, loc="lower center", ncol=len(legend_h),
                   bbox_to_anchor=(0.5, -0.025), fontsize=7.8, columnspacing=1.05,
                   handletextpad=0.5, handler_map={se_proxy: _WhiskerHandler()})
        fig.suptitle(title, fontsize=13, weight="bold", y=0.965)
    return fig


def save_all_with_degradation(results, outdir, degradation, scenarios=None,
                              fmts=("pdf", "png"), curves=None,
                              boundary=None, stem="mc_joint",
                              title="Simulation Study: Model Comparison (Joint)"):
    fig = composite_with_degradation(results, curves, degradation, scenarios, boundary,
                                     title=title)
    paths = []
    for ext in fmts:
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=400)
        paths.append(p)
    plt.close(fig)
    return paths

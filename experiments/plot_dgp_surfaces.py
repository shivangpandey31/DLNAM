"""Supplementary figure: true Monte Carlo DGP curves and 3D surfaces."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from dlnam_bench import plots as bp
from dlnam_sim import scenarios as sc
from dlnam_sim.scenarios import (
    LAG_MAX,
    REFERENCE,
    SCENARIO_DISPLAY_NAMES,
    SCENARIO_KEYS,
    SURFACE_FUNCTIONS,
    VALUE_RANGE,
)
from experiment_io import results_dir


HERE = Path(__file__).resolve().parent
RESULTS_DIR = results_dir(HERE)
OUT_STEM = "dgp_surfaces"

PLOT_RC = {
    **bp._RC,
    "axes.grid": False,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
}


def _lightened(colour: str, amount: float = 0.40) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(colour), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def _surface_facecolors(rr: np.ndarray, colour: str) -> np.ndarray:
    base = np.asarray(to_rgb(colour), dtype=float)
    light = np.asarray(_lightened(colour, 0.76), dtype=float)
    z = np.asarray(rr, dtype=float)
    lo, hi = np.nanpercentile(z, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros_like(z)
    else:
        scaled = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    scaled = scaled ** 0.75
    rgb = light[None, None, :] * (1.0 - scaled[..., None]) + base[None, None, :] * scaled[..., None]
    alpha = np.full((*z.shape, 1), 0.94)
    return np.concatenate([rgb, alpha], axis=-1)


def _padded_limits(values: np.ndarray, fraction: float = 0.035) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min())
    hi = float(finite.max())
    pad = fraction * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


INTEGER_LAGS = np.arange(0, int(LAG_MAX) + 1, dtype=float)


@contextmanager
def _lag_normalisation_pinned():
    """Let the DGPs be evaluated continuously in lag at their defined scale.

    `norm_decay` and `norm_bump` divide by a sum taken over whatever lag axis
    they are handed, so evaluating on a dense grid silently renormalises every
    kernel and shrinks the surface by the ratio of the grid sizes. The
    surfaces are defined with that constant fixed by the integer lags
    \\(\\ell'=0,\\dots,L\\), so we pin the denominator there and let only the
    numerator vary continuously. This is the function the supplement writes
    down, drawn as a smooth surface rather than sampled at integer lags.
    """
    def norm_decay(lag, scale):
        w = sc.pexp(-np.asarray(lag, dtype=float) / scale)
        return w / sc.pexp(-INTEGER_LAGS / scale).sum()

    def norm_bump(lag, center, width):
        w = sc.bump(np.asarray(lag, dtype=float), center, width)
        return w / sc.bump(INTEGER_LAGS, center, width).sum()

    originals = (sc.norm_decay, sc.norm_bump)
    sc.norm_decay, sc.norm_bump = norm_decay, norm_bump
    try:
        yield
    finally:
        sc.norm_decay, sc.norm_bump = originals


def _surface_values(key: str, values: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative log RR over the integer lags, and the continuous surface.

    `lags` must contain the integers 0..L, which is checked: the pinned
    surface has to agree with the untouched DGP everywhere the DGP is
    defined, or the figure is not showing the simulated truth.
    """
    fn = SURFACE_FUNCTIONS[key]
    reference = np.array([[REFERENCE]], dtype=float)

    def centred(ell: np.ndarray) -> np.ndarray:
        raw = np.asarray(fn(values[:, None], ell[None, :]), dtype=float)
        ref = np.asarray(fn(reference, ell[None, :]), dtype=float)
        return raw - ref

    exact = centred(INTEGER_LAGS)          # the truth, on its own lag grid
    with _lag_normalisation_pinned():
        dense = centred(lags)

    index = np.searchsorted(lags, INTEGER_LAGS)
    assert np.allclose(lags[index], INTEGER_LAGS), "lag grid must contain 0..L"
    assert np.allclose(dense[:, index], exact, atol=1e-10), \
        f"{key}: pinned surface disagrees with the DGP at integer lags"

    return exact.sum(axis=1), dense.T


def _clean_3d_axis(
    ax,
    *,
    first: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    ax.patch.set_alpha(0.0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-48)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_box_aspect((1.0, 1.0, 0.74))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_zticks(np.round(np.linspace(zlim[0], zlim[1], 3), 2))
    ax.tick_params(axis="both", which="major", pad=-2, length=2)
    ax.tick_params(axis="z", which="major", pad=-1, length=2)
    ax.set_xlabel("Exposure", labelpad=-1)
    ax.set_ylabel("Lag", labelpad=-2)
    ax.set_zlabel("RR" if first else "", labelpad=0)   # matches the Chicago figure
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((0.18, 0.18, 0.18, 1))
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["linewidth"] = 0.55


def plot_dgp_surfaces() -> list[Path]:
    values = np.linspace(VALUE_RANGE[0], VALUE_RANGE[1], 640)
    lags = np.linspace(0.0, float(LAG_MAX), 30 * int(LAG_MAX) + 1)  # contains 0..L
    xlim = _padded_limits(values, 0.055)
    ylim = _padded_limits(lags, 0.055)

    cumulative: dict[str, np.ndarray] = {}
    surfaces: dict[str, np.ndarray] = {}
    for key in SCENARIO_KEYS:
        cumulative[key], surfaces[key] = _surface_values(key, values, lags)

    curve_rr = {key: np.exp(cumulative[key]) for key in SCENARIO_KEYS}
    surface_rr = {key: np.exp(surfaces[key]) for key in SCENARIO_KEYS}
    curve_ymin = min(float(np.nanmin(v)) for v in curve_rr.values())
    curve_ymax = max(float(np.nanmax(v)) for v in curve_rr.values())
    curve_pad = 0.05 * max(curve_ymax - curve_ymin, 1e-9)
    curve_ylim = (max(0.0, curve_ymin - curve_pad), curve_ymax + curve_pad)

    with plt.rc_context(PLOT_RC):
        # Chicago's geometry, in inches, with one column dropped and no legend
        # row: box 2.654 wide by 2.817 high, 0.274 between columns, a 1.605
        # curve row, and the surface row overlapping it by 0.131 (a 3D axes
        # leaves internal margin the surface never reaches).
        fig = plt.figure(figsize=(12.27, 5.43))
        left, right, gap = 0.0557, 0.9876, 0.0223
        width = (right - left - gap * (len(SCENARIO_KEYS) - 1)) / len(SCENARIO_KEYS)
        curve_y, curve_h = 0.5597, 0.2956
        surface_y, surface_h = 0.0651, 0.5187
        curve_axes = []
        surface_axes = []
        for index in range(len(SCENARIO_KEYS)):
            x0 = left + index * (width + gap)
            curve_ax = fig.add_axes([x0, curve_y, width, curve_h])
            surface_ax = fig.add_axes([x0, surface_y, width, surface_h], projection="3d")
            curve_ax.set_zorder(5)
            curve_ax.patch.set_facecolor("white")
            curve_ax.patch.set_alpha(1.0)
            surface_ax.set_zorder(1)
            curve_axes.append(curve_ax)
            surface_axes.append(surface_ax)

        surface_colour = bp.COLOURS["DLNAM"]
        for index, key in enumerate(SCENARIO_KEYS):
            ax = curve_axes[index]
            rr_curve = curve_rr[key]
            ax.plot(values, rr_curve, color=bp.TRUTH_COLOUR, lw=1.25)
            ax.axhline(1.0, color="0.88", lw=0.7, zorder=0)
            ax.set_title(SCENARIO_DISPLAY_NAMES[key], fontsize=9, weight="bold", pad=6)
            ax.set_xlim(*VALUE_RANGE)
            ax.set_ylim(*curve_ylim)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel("Exposure")
            if index == 0:
                ax.set_ylabel("Cumulative RR")
            else:
                ax.tick_params(labelleft=False)

            ax3 = surface_axes[index]
            rr_surface = surface_rr[key]
            rr_min = float(np.nanmin(rr_surface))
            rr_max = float(np.nanmax(rr_surface))
            pad = 0.06 * max(rr_max - rr_min, 1e-9)
            zlim = (max(0.0, rr_min - pad), rr_max + pad)
            x_mesh, lag_mesh = np.meshgrid(values, lags)
            surface = ax3.plot_surface(
                x_mesh,
                lag_mesh,
                rr_surface,
                rstride=1,
                cstride=1,
                facecolors=_surface_facecolors(rr_surface, surface_colour),
                linewidth=0.0,
                edgecolor="none",
                antialiased=True,
                shade=False,
            )
            # ~270k quads; as vector art that is a 66 MB PDF, so rasterise the
            # surface itself and leave axes, ticks and text as vectors
            surface.set_rasterized(True)
            _clean_3d_axis(ax3, first=index == 0,
                           xlim=xlim, ylim=ylim, zlim=zlim)

        fig.suptitle("Simulation Study: Data-Generating Processes",
                     fontsize=13, weight="bold", x=0.5, y=0.9783)
        paths = [RESULTS_DIR / f"{OUT_STEM}.png", RESULTS_DIR / f"{OUT_STEM}.pdf"]
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=450)
        plt.close(fig)
    return paths


def main() -> None:
    paths = plot_dgp_surfaces()
    print("DGP 3D surface figure")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

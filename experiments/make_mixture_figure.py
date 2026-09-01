#!/usr/bin/env python3
"""Decompose fitted exposure-lag components into their subnetwork contributions.

One panel per simulation scenario, drawn in the same visual system as the
Monte Carlo cumulative-response panels: cumulative relative risk against the
exposure grid, boundary rules, the data-generating truth as a dense dotted
line, and one curve per component.

Each panel shows the fitted component (the mixture) together with the
individual contributions omega_s * f_s that sum to it. A single ensemble member
is fitted per scenario, so the decomposition is exactly f = sum_s omega_s f_s
with no averaging on top.

Output goes to experiments/results/ alongside the other result artefacts.

Usage:
    python experiments/make_mixture_figure.py [epochs]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dlnam_bench.plots as bp
from dlnam.config import (ActivationSpec, ExUSpec, InitSpec, LayerSpec,
                          ModelConfig, SurfaceTermSpec, TrainConfig)
from dlnam.data import DataProcessor
from dlnam.train import Trainer
from dlnam_sim.scenarios import (LAG_MAX, REFERENCE, VALUE_RANGE,
                                 gp_weather, scenarios)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCENARIOS = ["dgp1", "dgp2", "dgp3", "dgp4"]
N_OBS = 5000
N_GRID = 200
SEED = 0

# Match the shared Monte Carlo/real-data figure system.
PLOT_RC = bp._RC

# The mixture takes the near-black ink used for the DLNAM throughout. The
# subnetwork contributions take the burgundy-to-rose steps of the same system.
MIXTURE_COLOUR = bp.COLOURS["DLNAM"]
SUBNET_COLOURS = [bp.COLOURS["QAIC"], bp.COLOURS["QBIC"], bp.COLOURS["TDLNM"]]
CURVE_LW = 0.75
LEGEND_LW = 1.15


def fit_one(scenario, epochs, n_obs=N_OBS, seed=SEED, device=None):
    """The reference component architecture with the ensemble collapsed to a
    single member, so that the mixture decomposition is exact."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    tl = lambda: InitSpec(scheme="torch_linear")
    spec = SurfaceTermSpec(
        layers=[LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl())],
        num_subnets=3, scaling="minmax", lag_max=LAG_MAX,
        input_exu=ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                          weight_std=0.5, surface_strategy="concat",
                          bias_init=InitSpec(scheme="uniform", lo=0.0, hi=1.0)),
        mix_init=InitSpec(scheme="normal", mean=0.0, std=0.1))
    model_config = ModelConfig(terms={"x": spec}, link="log")
    train_config = TrainConfig(epochs=epochs, n_ensemble=1, lr=8e-4,
                               lr_min=1e-4, weight_decay=1e-4,
                               schedule="cosine", grad_clip=10, seed=seed)
    dgp = scenarios()[scenario]
    sim = dgp.simulate(n_obs, seed)
    trainer = Trainer(model_config, train_config, device=torch.device(device))
    prepared = DataProcessor(model_config).prepare(
        sim.frame, trainer.ensemble, dgp.target_col)
    trainer.fit(prepared.inputs, prepared.y)
    return trainer.ensemble[0].term("x"), dgp


def decompose(term, grid):
    """Cumulative relative risk of the mixture and of each contribution.

    Every curve is centred at the reference exposure, matching the centring of
    every effect reported in the paper, and returned on the relative-risk scale.
    """
    lags = term.lag_grid.detach().cpu().numpy()
    device = term.lag_grid.device
    xs = term._to_scaled(grid)
    ref = float(term._to_scaled(np.asarray([REFERENCE]))[0])

    def pairs(values):
        v = torch.tensor(np.repeat(values, len(lags)), dtype=torch.float32,
                         device=device).view(-1, 1)
        l = torch.tensor(np.tile(lags, len(values)), dtype=torch.float32,
                         device=device).view(-1, 1)
        return torch.cat([v, l], dim=1)

    vl, vl_ref = pairs(xs), pairs(np.full(len(xs), ref))
    weights = term._mix().detach().cpu().numpy()

    parts = []
    with torch.no_grad():
        for w, subnet in zip(weights, term.subnets):
            z = (subnet(vl) - subnet(vl_ref)).view(len(xs), len(lags))
            parts.append(np.exp(float(w) * z.cpu().numpy().sum(axis=1)))
        mixture = np.exp((term._mixed(vl) - term._mixed(vl_ref))
                         .view(len(xs), len(lags)).cpu().numpy().sum(axis=1))
    return mixture, parts, weights


def truth_curve(dgp, grid):
    lags = np.arange(LAG_MAX + 1, dtype=float)
    f = dgp.true_terms["x"].fn
    X = np.repeat(np.asarray(grid)[:, None], len(lags), axis=1)
    L = np.repeat(lags[None, :], len(grid), axis=0)
    Z = np.asarray(f(X, L))
    Z = Z - Z[int(np.argmin(np.abs(np.asarray(grid) - REFERENCE)))][None, :]
    return np.exp(Z.sum(axis=1))


def boundary_marks(seed=SEED, n_obs=N_OBS):
    x = gp_weather(n_obs, np.random.default_rng(seed))
    return float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))


def main(epochs=2500):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid = np.linspace(VALUE_RANGE[0], VALUE_RANGE[1], N_GRID)
    q_lo, q_hi = boundary_marks()

    panels = {}
    for scenario in SCENARIOS:
        print(f"[{scenario}] fitting")
        term, dgp = fit_one(scenario, epochs)
        mixture, parts, weights = decompose(term, grid)
        panels[scenario] = {"mixture": mixture, "parts": parts,
                            "truth": truth_curve(dgp, grid), "weights": weights}
        print(f"[{scenario}] mixing weights {np.round(weights, 3)}")

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13.5, 3.1),
                                 sharey=False)
        for scenario, ax in zip(SCENARIOS, np.atleast_1d(axes)):
            p = panels[scenario]
            ax.axhline(1.0, color="0.92", lw=0.5, zorder=0)

            n_curves = 1 + len(p["parts"])
            ax.plot(grid, p["mixture"], color=MIXTURE_COLOUR, lw=CURVE_LW,
                    zorder=2 + n_curves)
            displayed_parts = list(reversed(p["parts"]))
            for i, part in enumerate(displayed_parts):
                ax.plot(grid, part, color=SUBNET_COLOURS[i % len(SUBNET_COLOURS)],
                        lw=CURVE_LW, zorder=1 + n_curves - i)
            truth_line, = ax.plot(grid, p["truth"], color=bp.TRUTH_COLOUR,
                                  ls=bp.TRUTH_LINESTYLE, lw=bp.TRUTH_LW,
                                  zorder=3 + n_curves)
            truth_line.set_path_effects(bp.TRUTH_PATH_EFFECTS)

            ax.set_title(bp._nm(scenario), fontsize=9, weight="bold",
                         loc="center", pad=6)
            ax.set_xlabel("Exposure", fontsize=8.5)
            ax.margins(x=0)
            ax.tick_params(labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_zorder(20)
        np.atleast_1d(axes)[0].set_ylabel("Cumulative RR", fontsize=8.5)

        handles = [
            plt.Line2D([0], [0], color=bp.TRUTH_COLOUR,
                       ls=bp.TRUTH_LINESTYLE, lw=1.5, label="DGP"),
            plt.Line2D([0], [0], color=MIXTURE_COLOUR, lw=LEGEND_LW,
                       solid_capstyle="butt", label="Mixture DLNAM"),
        ]
        handles += [
            plt.Line2D([0], [0], color=SUBNET_COLOURS[i], lw=LEGEND_LW,
                       solid_capstyle="butt", label=f"Subnetwork {i + 1}")
            for i in range(len(SUBNET_COLOURS))
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.005), fontsize=7.8,
                   columnspacing=1.05, handletextpad=0.5)
        fig.suptitle("Mixture DLNAM: Subnetwork Decomposition",
                     fontsize=13, weight="bold", x=0.5, y=0.96)
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.27, top=0.78,
                            wspace=0.12)

        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(RESULTS_DIR, f"mixture_decomposition.{ext}"),
                bbox_inches="tight",
                dpi=400,
            )
        plt.close(fig)

    print("wrote", os.path.join(RESULTS_DIR, "mixture_decomposition.pdf"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2500)

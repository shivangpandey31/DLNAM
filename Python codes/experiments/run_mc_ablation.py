"""
run_mc_ablation.py -- Monte-Carlo ablation of DLNAM components, over the same DGP
surfaces as the model-comparison study. Reuses the MC harness and the composite
figure. The axis is {reference model, and one-component-changed variants}; each
variant differs from the reference by exactly ONE thing (ceteris paribus), so the
change in error/coverage is attributable to that component.

Ablations (one change each vs the reference):
    Reference    the complete DLNAM (ExU on, 3 subnets, smooth Mish, 3-ensemble)
    No ExU       input_exu.enabled = False        (linear input layer)
    No Subnets   num_subnets = 1                  (single shape-fn subnet; the
                 3-member ENSEMBLE is KEPT, so only the subnet mechanism changes)
    No Smooth    Mish -> ReLU-1 (Hardtanh(0,1))   (non-smooth, piecewise-linear
                 capped activation -- the NAM ExU pairing -- instead of a smooth
                 activation in the shape-function subnets)

Curves (panel A) show how each change distorts the recovered surface; the
bias^2/variance decomposition (panel B) attributes the cost to squared bias vs variance.
Output: mc_ablation.{pdf,png}.
"""
from __future__ import annotations

import os
import sys
# Make the project root and experiments/ importable without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dlnam.config import (ModelConfig, TrainConfig, SurfaceTermSpec, LayerSpec,
                          ExUSpec, ActivationSpec, InitSpec)
from dlnam.terms.base import Centering
from dlnam_sim.study import MonteCarloStudy
from dlnam_sim.scenarios import scenarios, LAG_MAX, VALUE_RANGE

from dlnam_bench import plots as bp
from experiment_io import results_dir, save_result_bundle
from run_mc import summarise

SCENARIOS = ["smooth", "delayed_peaks", "localized_peak", "tilting_threshold"]
N_REPS, N_OBS, EPOCHS, SEED = 3, 5000, 2500, 0
N_ENSEMBLE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF = 20.0

# palette / labels / markers: reuse the CANONICAL Slate->Ice sequence and markers
# from dlnam_bench.plots so every MC figure shares one visual language. Full model =
# darkest (the "hero", as DLNAM is in the main study); ablations lighten in order.
_SEQ = [bp.COLOURS[m] for m in ("DLNAM", "QAIC", "QBIC", "Penalised")]
_MK = ["o", "^", "s", "D"]                            # == plots.MARKERS order
ABLATIONS = ["reference", "no_exu", "no_subnets", "no_smooth"]
ABL_COLOURS = {k: _SEQ[i] for i, k in enumerate(ABLATIONS)}
ABL_MARKERS = {k: _MK[i] for i, k in enumerate(ABLATIONS)}
ABL_LABELS = {"reference": "Reference", "no_exu": "ExU Ablation",
              "no_subnets": "Mixture Ablation",
              "no_smooth": "Smoothness Ablation"}


def base_surface_spec(lag, *, exu=True, subnets=3, activation=None, mix_init="normal"):
    """Reference DLNAM surface spec. `activation` is an ActivationSpec factory for
    the shape-function subnet layers (default: smooth Mish). Penalty is kept at the
    reference value throughout (this ablation set does not vary it)."""
    act = activation if activation is not None else (lambda: ActivationSpec(base=torch.nn.Mish))
    mix_init = None if mix_init is None else InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl = lambda: InitSpec(scheme="torch_linear")
    input_exu = (ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                         weight_std=0.5, surface_strategy="concat",
                         bias_init=exu_bias()) if exu else
                 ExUSpec(enabled=False))
    return SurfaceTermSpec(
        layers=[LayerSpec(128, act()),
                LayerSpec(128, act(), weight_init=tl(), bias_init=tl())],
        num_subnets=subnets, scaling="minmax", lag_max=lag,
        input_exu=input_exu, mix_init=mix_init)


def config_for(key, lag):
    """Return (ModelConfig, n_ensemble) for an ablation key -- one change each.
    The 3-member ensemble is kept for every variant (including No Subnets);
    only the named component changes."""
    relu1 = lambda: ActivationSpec(base=lambda: torch.nn.Hardtanh(0.0, 1.0))  # ReLU-1
    if key == "reference":
        spec = base_surface_spec(lag)
    elif key == "no_exu":
        spec = base_surface_spec(lag, exu=False)
    elif key == "no_subnets":
        spec = base_surface_spec(lag, subnets=1, mix_init=None)           # ensemble stays at 3
    elif key == "no_smooth":
        spec = base_surface_spec(lag, activation=relu1)    # Mish -> ReLU-1
    else:
        raise ValueError(key)
    return ModelConfig(terms={"x": spec}, link="log"), N_ENSEMBLE


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    scen = scenarios(lag_max=LAG_MAX)
    grid = np.linspace(*VALUE_RANGE, 200)
    cen = Centering(method="reference", value=REF)

    results, curves, boundary, timing = {}, {}, {}, {}
    for s in SCENARIOS:
        dgp = scen[s]
        xvals = dgp.simulate(N_OBS, SEED).frame["x"].values
        q_lo, q_hi = np.quantile(xvals, 0.05), np.quantile(xvals, 0.95)
        boundary[s] = (float(q_lo), float(q_hi))
        m_bnd = (grid < q_lo) | (grid > q_hi); m_int = ~m_bnd

        row, cv, tm = {}, None, {}
        for key in ABLATIONS:
            mcfg, n_ens = config_for(key, LAG_MAX)
            study = MonteCarloStudy(
                dgp=dgp, model_config=mcfg,
                train_config=TrainConfig(epochs=EPOCHS, n_ensemble=n_ens,
                                         lr=8e-4, lr_min=1e-4, weight_decay=1e-4,
                                         schedule="cosine", grad_clip=10, seed=SEED),
                centering=cen, n_reps=N_REPS, n_obs=N_OBS, base_seed=SEED,
                se_source="laplace", device=DEVICE)
            st = study.run(progress=True)
            row[key] = summarise(st, m_int, m_bnd)
            tm[key] = st.timing_summary()
            if cv is None:
                cv = {"grid": np.asarray(st.grids["x"]),
                      "truth": np.asarray(st.truth["x"])}
            cv[key] = np.asarray(st._stack("x", "mean").mean(0))
            print(f"[{s:18s}] {ABL_LABELS[key]:14s} err {row[key]['err_tot']:.4f}"
                  f" ± {row[key]['err_tot_se']:.4f} bias^2 {row[key]['bias2_tot']:.2e}"
                  f" var {row[key]['var_tot']:.2e} cov {row[key]['cov_tot']:.2f}")
        results[s] = row
        curves[s] = cv
        timing[s] = tm

    models = list(ABLATIONS)
    settings = {
        "n_reps": N_REPS,
        "n_obs": N_OBS,
        "epochs": EPOCHS,
        "n_ensemble": N_ENSEMBLE,
        "lag": LAG_MAX,
        "reference": REF,
        "seed": SEED,
        "device": DEVICE,
        "value_range": list(VALUE_RANGE),
        "se_source": "laplace",
        "labels": ABL_LABELS,
    }
    out_dir = results_dir(here)
    result_path = out_dir / "mc_ablation.json"
    save_result_bundle(
        result_path,
        kind="dlnam_architecture_ablation_mc",
        settings=settings,
        scenarios=SCENARIOS,
        models=models,
        results=results,
        boundary=boundary,
        curves=curves,
        timing=timing,
    )
    print(f"saved {result_path}")
    _co, _mk, _la, _mo = bp.COLOURS, bp.MARKERS, bp.LABELS, bp.MODELS
    bp.COLOURS, bp.MARKERS, bp.LABELS = ABL_COLOURS, ABL_MARKERS, ABL_LABELS
    bp.MODELS = list(ABLATIONS)
    try:
        paths = bp.save_all(results, out_dir, scenarios=SCENARIOS,
                            curves=curves, boundary=boundary, stem="mc_ablation",
                            title="Simulation Study: Architecture Ablation")
        for p in paths:
            print(f"saved {p}")
    finally:
        bp.COLOURS, bp.MARKERS, bp.LABELS, bp.MODELS = _co, _mk, _la, _mo


if __name__ == "__main__":
    main()

"""
run_mc_exu.py -- Monte-Carlo evaluation of the three ExU surface-encoder
generalization strategies, over the same DGP surfaces as the model-comparison
study. Reuses dlnam_sim.MonteCarloStudy (harness) and dlnam_bench.plots
(composite figure). The "models" axis here is the three ExU strategies, not
DLNAM/DLNM.

The three strategies (dlnam.config.SurfaceEncoderStrategy):
    concat               per-dimension scalar ExU, concatenated (baseline)
    unified_shared_bias  one bias per input dim (no per-unit localisation)
    unified_local_bias   bias per unit and per dim

Everything else (layers, subnets, penalty, training) is held FIXED so the
comparison is ceteris paribus -- only surface_strategy changes. Bias^2/variance
decomposition is the point: an under-localised strategy pays BIAS at sharp
features; an over-flexible one pays VARIANCE. Curves (panel A) give intuition for
which strategy tracks each true surface.

Output: mc_exu.{pdf,png} via dlnam_bench.plots.save_all (composite: A curves,
B error, C coverage).
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
from run_mc import summarise   # identical region-summary contract

# --- experiment axis: the three ExU strategies -----------------------------
STRATEGIES = [
    ("concat",              "Concatenation"),
    ("unified_shared_bias", "Unified Shared Bias"),
    ("unified_local_bias",  "Unified Local Bias"),
]
# palette / labels for the composite: reuse the CANONICAL Slate->Ice sequence and
# markers from dlnam_bench.plots so every MC figure shares one visual language. The
# three ExU strategies map onto the first three sequence colours (darkest->light).
_SEQ = [bp.COLOURS[m] for m in ("DLNAM", "QAIC", "QBIC", "Penalised")]
_MK = ["o", "^", "s", "D"]                            # == plots.MARKERS order
_KEYS = [k for k, _ in STRATEGIES]
EXU_COLOURS = {k: _SEQ[i] for i, k in enumerate(_KEYS)}
EXU_MARKERS = {k: _MK[i] for i, k in enumerate(_KEYS)}
EXU_LABELS = {k: lbl for k, lbl in STRATEGIES}

SCENARIOS = ["smooth", "delayed_peaks", "localized_peak", "tilting_threshold"]
N_REPS, N_OBS, EPOCHS, N_ENSEMBLE, SEED = 3, 5000, 2500, 3, 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF = 20.0


def model_config(strategy: str, lag: int) -> ModelConfig:
    """DLNAM surface config identical to the main study except surface_strategy."""
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)
    tl = lambda: InitSpec(scheme="torch_linear")
    return ModelConfig(terms={"x": SurfaceTermSpec(
        layers=[LayerSpec(128, mish()),
                LayerSpec(128, mish(), weight_init=tl(), bias_init=tl())],
        num_subnets=3, scaling="minmax", lag_max=lag,
        input_exu=ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                          weight_std=0.5, surface_strategy=strategy,
                          bias_init=exu_bias()),
        mix_init=mix_init())}, link="log")


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
        m_bnd = (grid < q_lo) | (grid > q_hi)
        m_int = ~m_bnd

        row, cv, tm = {}, None, {}
        for strat, _ in STRATEGIES:
            study = MonteCarloStudy(
                dgp=dgp, model_config=model_config(strat, LAG_MAX),
                train_config=TrainConfig(epochs=EPOCHS, n_ensemble=N_ENSEMBLE,
                                         lr=8e-4, lr_min=1e-4, weight_decay=1e-4,
                                         schedule="cosine", grad_clip=10, seed=SEED),
                centering=cen, n_reps=N_REPS, n_obs=N_OBS, base_seed=SEED,
                se_source="laplace", device=DEVICE)
            st = study.run(progress=True)
            row[strat] = summarise(st, m_int, m_bnd)
            tm[strat] = st.timing_summary()
            # curves: truth once + MC-mean per strategy
            if cv is None:
                cv = {"grid": np.asarray(st.grids["x"]),
                      "truth": np.asarray(st.truth["x"])}
            cv[strat] = np.asarray(st._stack("x", "mean").mean(0))
            print(f"[{s:18s}] {strat:20s} err {row[strat]['err_tot']:.4f}"
                  f" ± {row[strat]['err_tot_se']:.4f} bias^2 {row[strat]['bias2_tot']:.2e}"
                  f" var {row[strat]['var_tot']:.2e} cov {row[strat]['cov_tot']:.2f}")
        results[s] = row
        curves[s] = cv
        timing[s] = tm

    models = [k for k, _ in STRATEGIES]
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
        "labels": EXU_LABELS,
    }
    out_dir = results_dir(here)
    result_path = out_dir / "mc_exu.json"
    save_result_bundle(
        result_path,
        kind="dlnam_exu_encoder_mc",
        settings=settings,
        scenarios=SCENARIOS,
        models=models,
        results=results,
        boundary=boundary,
        curves=curves,
        timing=timing,
    )
    print(f"saved {result_path}")
    # reuse the composite, but relabel the model axis to the ExU strategies
    _co, _mk, _la, _mo = bp.COLOURS, bp.MARKERS, bp.LABELS, bp.MODELS
    bp.COLOURS, bp.MARKERS, bp.LABELS = EXU_COLOURS, EXU_MARKERS, EXU_LABELS
    bp.MODELS = [k for k, _ in STRATEGIES]
    try:
        paths = bp.save_all(results, out_dir, scenarios=SCENARIOS,
                            curves=curves, boundary=boundary, stem="mc_exu",
                            title="Simulation Study: ExU Encoder Comparison")
        for p in paths:
            print(f"saved {p}")
    finally:
        bp.COLOURS, bp.MARKERS, bp.LABELS, bp.MODELS = _co, _mk, _la, _mo


if __name__ == "__main__":
    main()

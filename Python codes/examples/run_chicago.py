"""
run_chicago.py — real-data run on the Chicago NMMAPS dataset.

The clean-architecture replacement for the original run_dlnam.py. Dataset-
specific feature engineering (the 2-day rolling-average pollutant columns) is
done HERE, upstream of the general DataProcessor, which then windows/scales/
encodes by term spec. Point CSV_PATH at your file and tune the SETTINGS block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from dlnam import (ModelConfig, TrainConfig, LayerSpec, ActivationSpec, ExUSpec,
                   SoftCapSpec, InitSpec, Trainer, DataProcessor, PerformanceEvaluator,
                   ResultVisualizer, WaldUQ, make_link, Centering)
from dlnam.config import (SurfaceTermSpec, SmoothTermSpec, TrendTermSpec,
                          CategoricalTermSpec)

# ----------------------------- SETTINGS -----------------------------------
CSV_PATH   = r"chicago_nmmaps.csv"   # <-- your path
LAG_MAX    = 30
EPOCHS     = 3500
N_ENSEMBLE = 3
SEED       = 123
DEVICE     = "cuda"
DOW_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]
# --------------------------------------------------------------------------

LABELS = {"temp": "Temperature", "dptp01": "Dew Point", "o301": "Ozone",
          "pm1001": "PM10", "trend": "Time", "dow": "Day of Week"}


def load_chicago(path):
    df = pd.read_csv(path)
    for col in ["dptp", "o3", "pm10"]:                    # 2-day rolling averages
        df[f"{col}01"] = df[col].rolling(window=2).mean()
    return df


def build_config():
    # Soft-capped activations for the ExU INPUT layers (bound the ExU output at
    # its source); plain activations on the hidden tail.
    softcapmish = lambda: ActivationSpec(base=torch.nn.Mish,
                                transform=SoftCapSpec(init_cap=0.5, learnable=True))
    softcapsilu = lambda: ActivationSpec(base=torch.nn.SiLU,
                                transform=SoftCapSpec(init_cap=0.5, learnable=True))
    mish = lambda: ActivationSpec(base=torch.nn.Mish)
    silu = lambda: ActivationSpec(base=torch.nn.SiLU)

    S = 3                                            # subnets per term
    # InitSpec used to state the defaults explicitly (change these to experiment):
    mix_init = lambda: InitSpec(scheme="normal", mean=0.0, std=0.1)   # = default 1/S
    exu_bias = lambda: InitSpec(scheme="uniform", lo=0.0, hi=1.0)    # = ExU bias default
    tl       = lambda: InitSpec(scheme="torch_linear")              # = PyTorch Linear default
    # (ExU input layers take their weight init from weight_mean/weight_std, so a
    # LayerSpec weight_init there would be ignored; only the linear layers below
    # consume tl(). The final scalar-output Linear keeps the PyTorch default.)

    return ModelConfig(terms={
        "temp": SurfaceTermSpec(
            layers=[LayerSpec(128, mish()),     # softcap on the ExU input layer
                    LayerSpec(128, mish(), weight_init=tl(), bias_init=tl())],  # plain tail
            num_subnets=S, scaling="minmax", lag_max=LAG_MAX,
            input_exu=ExUSpec(enabled=True, weight_mean=1.5, weight_mean_lag=2.5,
                              weight_std=0.5, surface_strategy="split_concat",
                              bias_init=exu_bias()),
            mix_init=mix_init()),                      # convex combo: constrain_subnet_weights=True
        "dptp01": SmoothTermSpec(layers=[LayerSpec(32, silu(), weight_init=tl(), bias_init=tl())],
                                 num_subnets=S, scaling="zscore", mix_init=mix_init()),
        "o301":   SmoothTermSpec(layers=[LayerSpec(32, silu(), weight_init=tl(), bias_init=tl())],
                                 num_subnets=S, scaling="zscore", mix_init=mix_init()),
        "pm1001": SmoothTermSpec(layers=[LayerSpec(32, silu(), weight_init=tl(), bias_init=tl())],
                                 num_subnets=S, scaling="zscore", mix_init=mix_init()),
        "trend":  TrendTermSpec(
            layers=[LayerSpec(128, silu()),     # softcap on the ExU input layer
                    LayerSpec(128, silu(), weight_init=tl(), bias_init=tl()),
                    LayerSpec(128, silu(), weight_init=tl(), bias_init=tl())],  # 3 layers, plain tail
            num_subnets=S,
            input_exu=ExUSpec(enabled=True, weight_mean=4.5, weight_std=0.5,
                              bias_init=exu_bias()),
            mix_init=mix_init()),
        "dow":    CategoricalTermSpec(num_categories=7, order=DOW_ORDER),  # no subnets/ExU
    }, link="log")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = load_chicago(CSV_PATH)
    mcfg = build_config()
    trainer = Trainer(mcfg, TrainConfig(epochs=EPOCHS, n_ensemble=N_ENSEMBLE,
                      lr=5e-4, lr_min=1e-4, weight_decay=1e-4, schedule="cosine",
                      seed=SEED), device=torch.device(DEVICE))
    proc = DataProcessor(mcfg)
    prepared = proc.prepare(df, trainer.ensemble, target_col="death")
    print(f"prepared {prepared.n_samples} samples; terms: {list(prepared.inputs)}")

    trainer.fit(prepared.inputs, prepared.y)

    ev = PerformanceEvaluator(trainer.ensemble, distribution="poisson")
    ev.report(ev.evaluate(prepared.inputs, prepared.y))

    link = make_link("log")
    cen = Centering(method="median")
    viz = ResultVisualizer(trainer.ensemble, link, WaldUQ("ensemble"), cen,
                           distribution="poisson", labels=LABELS, trainer=trainer)

    # 2-panel figure for consistency with the dgps outputs: cumulative RR + contour.
    # (The model still fits all terms -- confounders, trend, dow -- for adjustment;
    # only the temperature exposure-response is plotted.)
    fig, (ax_temp, ax_cont) = plt.subplots(1, 2, figsize=(13, 5))
    viz.plot_effect("temp", ax=ax_temp)        # cumulative RR vs temperature
    viz.plot_surface("temp", ax=ax_cont)       # lag surface contour

    fig.suptitle("Chicago NMMAPS — DLNAM", fontsize=14)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chicago_dlnam.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

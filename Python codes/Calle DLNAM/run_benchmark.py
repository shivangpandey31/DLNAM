# run_benchmark.py
#
# Runs multiple DLNAM configurations sequentially on the Chicago NMMAPS
# dataset and collects their metrics for comparison.
#
# Usage:
#   1. Define a base configuration in base_cfg()
#   2. Add experiment configurations in define_experiments()
#      — each experiment overrides only the keys it changes
#   3. Run: python run_benchmark.py
#
# Results are printed as a summary table at the end.
# Trained trainers and metrics are returned from main() for downstream
# plotting and analysis (to be implemented in run_benchmark_plots.py).

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from DLNAM.data_utils    import DLNAMDataProcessor
from DLNAM.models        import Multilayer_DLNAM
from DLNAM.train         import Trainer
from DLNAM.visualization import ResultVisualizer
from DLNAM.evaluation    import PerformanceEvaluator
from run_dlnam           import load_chicago, run_model


# ==========================================================================
# BASE CONFIGURATION
# All experiments inherit from this. Override only what changes.
# ==========================================================================

def base_cfg(device):
    return dict(

        # --- device ---
        device             = device,
        use_compile        = False,

        # --- ensemble ---
        n_ensemble         = 3,
        n_subnets          = 3,

        # --- architecture ---
        surface_layers     = [128, 128],
        conf_layers        = [32],
        trend_layers       = [256, 128, 64],

        # --- activations ---
        surface_activation = nn.Mish,
        trend_activation   = lambda: nn.Hardtanh(0, 1),
        conf_activation    = nn.Mish,

        # --- ExU ---
        use_exu_exposure   = True,
        use_exu_trend      = True,
        use_exu_lag        = True,
        exu_mean_val       = 1.5,
        exu_mean_trend     = 3.5,
        exu_mean_lag       = 2.5,

        # --- mixing weights ---
        # True  -> softmax convex combination (standard DLNAM at S=1)
        # False -> unconstrained raw weights
        constrain_weights  = False,

        # --- regularisation ---
        dropout_p          = 0.0,



        subnet_dropout_p   = 0.0,
        output_penalty     = 0,
        weight_decay       = 0.0,

        # --- day of week ---
        use_dow            = True,

        # --- training ---
        epochs             = 3000,
        batch_fraction     = None,
        lr_schedule        = 'cosine',
        lr                 = 5e-4,
        lr_min             = 3e-4,
        lr_plateau_factor  = 0.5,
        lr_plateau_patience= 10,

        # --- uncertainty ---
        alpha              = 0.05,
        ci_type            = 'ensemble',
        centering          = 'median',
    )


# ==========================================================================
# EXPERIMENTS
# Each entry is a dict with:
#   'name'      : short label used in the results table
#   'overrides' : dict of keys to override in base_cfg
# ==========================================================================

def define_experiments(device):
    base = base_cfg(device)

    experiments = [

        # ------------------------------------------------------------------
        # Baseline — standard configuration
        # ------------------------------------------------------------------
        {
            'name'     : 'Baseline',
            'overrides': {},
        },

        # ------------------------------------------------------------------
        # Constrained vs unconstrained mixing weights
        # ------------------------------------------------------------------
        {
            'name'     : 'Constrained weights (S=3)',
            'overrides': {'constrain_weights': True},
        },

        # ------------------------------------------------------------------
        # Number of sub-networks S
        # ------------------------------------------------------------------
        {
            'name'     : 'S=1 (standard DLNAM)',
            'overrides': {'n_subnets': 1},
        },
        {
            'name'     : 'S=5',
            'overrides': {'n_subnets': 5},
        },

        # ------------------------------------------------------------------
        # Network width
        # ------------------------------------------------------------------
        {
            'name'     : 'Width 32',
            'overrides': {'surface_layers': [32, 32]},
        },
        {
            'name'     : 'Width 64',
            'overrides': {'surface_layers': [64, 64]},
        },
        {
            'name'     : 'Width 256',
            'overrides': {'surface_layers': [256, 256]},
        },

        # ------------------------------------------------------------------
        # Network depth
        # ------------------------------------------------------------------
        {
            'name'     : 'Depth 1',
            'overrides': {'surface_layers': [128]},
        },
        {
            'name'     : 'Depth 3',
            'overrides': {'surface_layers': [128, 128, 128]},
        },

        # ------------------------------------------------------------------
        # Learning rate schedule
        # ------------------------------------------------------------------
        {
            'name'     : 'No schedule',
            'overrides': {'lr_schedule': False},
        },
        {
            'name'     : 'Plateau schedule',
            'overrides': {'lr_schedule': 'plateau'},
        },

        # ------------------------------------------------------------------
        # Weight decay
        # ------------------------------------------------------------------
        {
            'name'     : 'Weight decay 1e-4',
            'overrides': {'weight_decay': 1e-4},
        },
        {
            'name'     : 'Weight decay 1e-3',
            'overrides': {'weight_decay': 1e-3},
        },

    ]

    # Merge each experiment's overrides into a full cfg
    configs = []
    for exp in experiments:
        cfg = copy.deepcopy(base)
        cfg.update(exp['overrides'])
        configs.append({'name': exp['name'], 'cfg': cfg})

    return configs


# ==========================================================================
# RESULTS TABLE
# ==========================================================================

def print_results_table(results):
    """
    Print a summary table of all experiment results.

    results : list of dicts with keys 'name' and 'metrics'
    """
    header = (f"{'Experiment':<35} {'R²':>8} {'RMSE':>8} {'Null RMSE':>10}"
              f" {'MAE':>8} {'Null MAE':>10}")
    print("\n" + "=" * len(header))
    print("BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        m = r['metrics']
        print(
            f"{r['name']:<35}"
            f" {m.get('R2',        float('nan')):>8.4f}"
            f" {m.get('RMSE',      float('nan')):>8.3f}"
            f" {m.get('Null_RMSE', float('nan')):>10.3f}"
            f" {m.get('MAE',       float('nan')):>8.3f}"
            f" {m.get('Null_MAE',  float('nan')):>10.3f}"
        )
    print("=" * len(header))


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("CUDA Available:   ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data — loaded once, shared across all experiments
    # ------------------------------------------------------------------
    CSV_PATH      = r"C:\Users\calle\OneDrive - KTH\Master's Thesis\DLNAM\chicago_nmmaps.csv"
    EXPOSURE_COLS = ['temp']
    LAG_MAX_LIST  = [30]
    CONF_COLS     = ['dptp01', 'o301', 'pm1001']

    data, processor, X_exposures, X_c, X_time, Y, X_dow = load_chicago(
        csv_path              = CSV_PATH,
        exposure_cols         = EXPOSURE_COLS,
        lag_max_list          = LAG_MAX_LIST,
        conf_cols             = CONF_COLS,
        scaling_type_exposure = 'minmax',
        scaling_type_conf     = 'zscore',
        use_dow               = True,
    )

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------
    experiments = define_experiments(device)

    # Set to None to run all, or e.g. ['Baseline', 'S=1 (standard DLNAM)']
    # to run a subset
    RUN_ONLY = None

    results  = []
    trainers = {}

    for exp in experiments:
        name = exp['name']
        cfg  = exp['cfg']

        if RUN_ONLY is not None and name not in RUN_ONLY:
            print(f"\n  Skipping: {name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Experiment: {name}")
        print(f"{'=' * 60}")

        trainer, metrics = run_model(
            data, processor, X_exposures, X_c, X_time, Y, X_dow,
            exposure_cols = EXPOSURE_COLS,
            conf_cols     = CONF_COLS,
            cfg           = cfg,
        )

        results.append({'name': name, 'metrics': metrics})
        trainers[name] = trainer

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_results_table(results)

    return trainers, results


if __name__ == '__main__':
    main()
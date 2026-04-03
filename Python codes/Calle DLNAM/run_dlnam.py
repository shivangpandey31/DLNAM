# run_dlnam.py
#
# Entry point for a single DLNAM experiment on the Chicago NMMAPS dataset.
# Everything is controlled through the single cfg dict in main().
#
#   load_chicago(cfg)           — loads and preprocesses data
#   run_model(..., cfg)         — trains, evaluates and visualises one ensemble
#   main()                      — defines cfg and calls the above

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from DLNAM.data_utils    import DLNAMDataProcessor
from DLNAM.models        import Multilayer_DLNAM
from DLNAM.train         import Trainer
from DLNAM.visualization import ResultVisualizer
from DLNAM.evaluation    import PerformanceEvaluator


# ==========================================================================
# DATA
# ==========================================================================

def load_chicago(cfg):
    """
    Load and preprocess the Chicago NMMAPS dataset.
    All settings are read from cfg.

    Returns
    -------
    data         : raw DataFrame (for visualisation)
    processor    : fitted DLNAMDataProcessor
    X_exposures  : list of lag tensors, one per exposure
    X_c          : confounder tensor
    X_time       : normalised time tensor
    Y            : target tensor (death counts)
    X_encodings  : list of LongTensors, one per entry in cfg['encoding_configs']
    """
    data = pd.read_csv(cfg['csv_path'])

    # 2-day rolling average for pollutants (matches R's filter(x, c(1,1)/2))
    for col in ['dptp', 'o3', 'pm10']:
        data[f'{col}01'] = data[col].rolling(window=2).mean()

    enc_cols  = [ec['col'] for ec in cfg.get('encoding_configs', [])]
    drop_cols = cfg['conf_cols'] + cfg['exposure_cols'] + enc_cols
    data      = data.dropna(subset=drop_cols + ['death']).reset_index(drop=True)

    processor = DLNAMDataProcessor()
    X_exposures, X_c, X_time, Y, X_encodings = \
        processor.prepare_agnostic_tensors(
            df                    = data,
            exposure_cols         = cfg['exposure_cols'],
            lag_max_list          = cfg['lag_max_list'],
            conf_cols             = cfg['conf_cols'],
            target_col            = 'death',
            scaling_type_exposure = cfg['scaling_type_exposure'],
            scaling_type_conf     = cfg['scaling_type_conf'],
            scaling_type_trend    = cfg['scaling_type_trend'],
            scaling_type_lag      = cfg['scaling_type_lag'],
            encoding_configs      = cfg.get('encoding_configs', []),
        )

    return data, processor, X_exposures, X_c, X_time, Y, X_encodings


# ==========================================================================
# MODEL
# ==========================================================================

def run_model(data, processor, X_exposures, X_c, X_time, Y, X_encodings, cfg):
    """
    Initialise, train, evaluate and visualise one DLNAM ensemble.
    All settings are read from cfg.

    Returns
    -------
    trainer : trained Trainer object
    metrics : dict of evaluation metrics
    """
    surf_configs = [cfg['surface_layers']] * len(cfg['exposure_cols'])
    conf_configs = [cfg['conf_layers']]    * len(cfg['conf_cols'])

    trainer = Trainer(
        Multilayer_DLNAM,
        # --- ensemble ---
        num_models         = cfg['n_ensemble'],
        # --- data dimensions ---
        exposure_dims      = [x.shape[1] for x in X_exposures],
        conf_dim           = len(cfg['conf_cols']),
        # --- network configs ---
        surface_configs    = surf_configs,
        conf_configs       = conf_configs,
        trend_layers       = cfg['trend_layers'],
        # --- subnets ---
        num_subnets        = cfg['n_subnets'],
        # --- activations ---
        surface_activation = cfg['surface_activation'],
        trend_activation   = cfg['trend_activation'],
        conf_activation    = cfg['conf_activation'],
        # --- ExU ---
        use_exu_exposure   = cfg['use_exu_exposure'],
        use_exu_trend      = cfg['use_exu_trend'],
        use_exu_lag        = cfg['use_exu_lag'],
        exu_mean_val       = cfg['exu_mean_val'],
        exu_mean_trend     = cfg['exu_mean_trend'],
        exu_mean_lag       = cfg['exu_mean_lag'],
        # --- regularisation ---
        dropout_p          = cfg['dropout_p'],
        subnet_dropout_p   = cfg['subnet_dropout_p'],
        output_penalty     = cfg['output_penalty'],
        # --- mixing weights ---
        constrain_weights  = cfg['constrain_weights'],
        # --- categorical encodings ---
        # Inject encoding_layers into each encoding config so the
        # architecture is fully controlled from the top-level cfg.
        encoding_configs   = [
            {**ec,
             'hidden_layers': ec.get('hidden_layers',
                                     cfg.get('encoding_layers', []))}
            for ec in cfg.get('encoding_configs', [])
        ],
        # --- device ---
        use_compile        = cfg['use_compile'],
        device             = cfg['device'],
    )

    trainer.train(
        X_exposures, X_c, X_time, Y,
        epochs              = cfg['epochs'],
        batch_fraction      = cfg['batch_fraction'],
        lr_schedule         = cfg['lr_schedule'],
        lr_min              = cfg['lr_min'],
        lr_plateau_factor   = cfg['lr_plateau_factor'],
        lr_plateau_patience = cfg['lr_plateau_patience'],
        optim_kwargs        = {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        x_encodings         = X_encodings if X_encodings else None,
        processor           = processor,
    )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    viz = ResultVisualizer(
        trainer, processor,
        alpha     = cfg['alpha'],
        ci_type   = cfg['ci_type'],
        centering = cfg['centering'],
    )
    viz.plot_all(
        data,
        exposure_cols    = cfg['exposure_cols'],
        conf_cols        = cfg['conf_cols'],
        Y                = Y,
        X_exposures      = X_exposures,
        X_c              = X_c,
        X_time           = X_time,
        X_encodings      = X_encodings if X_encodings else None,
        encoding_configs = cfg.get('encoding_configs', []),
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluator = PerformanceEvaluator(trainer)
    metrics   = evaluator.calculate_metrics(
        X_exposures, X_c, X_time, Y,
        alpha        = cfg['alpha'],
        ci_type      = cfg['ci_type'],
        x_encodings  = X_encodings if X_encodings else None,
    )
    evaluator.print_report(metrics)

    return trainer, metrics


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("CUDA Available:   ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(

        # --- paths and columns ---
        csv_path      = r"C:\Users\calle\OneDrive - KTH\Master's Thesis\DLNAM\chicago_nmmaps.csv",
        exposure_cols = ['temp'],
        lag_max_list  = [30],
        conf_cols     = ['dptp01', 'o301', 'pm1001'],

        # --- categorical encodings ---
        # List of dicts, one per categorical variable to include.
        # Each dict must contain:
        #   'name'           : str  — label used in diagnostics
        #   'num_categories' : int  — number of distinct levels
        #   'col'            : str  — DataFrame column name
        #   'order'          : list — category names in desired integer order;
        #                             index 0 is the reference level (absorbed
        #                             into the global intercept, analogous to
        #                             R's treatment contrasts)
        # Add more dicts to include additional categorical variables,
        # e.g. month-of-year, season, city, etc.
        encoding_configs = [
            {
                'name'           : 'dow',
                'num_categories' : 7,
                'col'            : 'dow',
                'order'          : ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                    'Friday', 'Saturday', 'Sunday'],
                # hidden_layers and activation are injected automatically
                # from encoding_layers below — no need to set them here
            },
        ],

        # --- scaling ---
        # 'minmax' -> [0,1] (recommended for ExU units)
        # 'zscore' -> zero mean, unit variance
        # trend and lag are always [0,1] by construction
        scaling_type_exposure = 'minmax',
        scaling_type_conf     = 'zscore',
        scaling_type_trend    = 'minmax',
        scaling_type_lag      = 'minmax',

        # --- device ---
        device      = device,
        use_compile = False,

        # --- ensemble ---
        n_ensemble = 3,
        n_subnets  = 3,

        # --- architecture ---
        # Hidden layer widths for each network type.
        # surface:  exposure-lag surface sub-networks (2D input: value × lag)
        # conf:     confounder sub-networks (1D input per confounder)
        # trend:    long-term trend network (1D input: normalised time)
        # encoding: one-hot encoding networks ([] = pure lookup table)
        surface_layers  = [128, 128],
        conf_layers     = [32],
        trend_layers    = [128, 128, 128],
        encoding_layers = [],

        # --- activations ---
        surface_activation = nn.Mish,
        trend_activation   = lambda: nn.Hardtanh(0, 1),
        conf_activation    = nn.Mish,

        # --- ExU ---
        use_exu_exposure = True,
        use_exu_trend    = True,
        use_exu_lag      = True,
        exu_mean_val     = 1.25,
        exu_mean_trend   = 3.5,
        exu_mean_lag     = 2.5,

        # --- mixing weights ---
        constrain_weights = False,

        # --- regularisation ---
        dropout_p        = 0.0,
        subnet_dropout_p = 0.0,
        output_penalty   = 0.05,
        weight_decay     = 0.0,

        # --- training ---
        epochs         = 3000,
        batch_fraction = None,
        lr_schedule    = 'cosine',
        lr             = 5e-4,
        lr_min         = 2.5e-4,
        lr_plateau_factor   = 0.5,
        lr_plateau_patience = 10,

        # --- uncertainty ---
        alpha     = 0.05,
        ci_type   = 'ensemble',
        centering = 'median',
    )

    data, processor, X_exposures, X_c, X_time, Y, X_encodings = load_chicago(cfg)

    trainer, metrics = run_model(
        data, processor, X_exposures, X_c, X_time, Y, X_encodings, cfg
    )


if __name__ == '__main__':
    main()
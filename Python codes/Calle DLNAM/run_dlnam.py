# run_dlnam.py

import pandas as pd
import torch
import torch.nn as nn

from DLNAM.data_utils import DLNAMDataProcessor
from DLNAM.models import Multilayer_DLNAM
from DLNAM.train import Trainer
from DLNAM.visualization import ResultVisualizer
from DLNAM.evaluation import PerformanceEvaluator

# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device Name: ", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 2500
MODELS_IN_ENSEMBLE = 3
NUM_SUBNETS = 3
LOSS_TYPE = 'Poisson'
EXPOSURE_SCALING = 'minmax'   # 'minmax' (0,1)  or  'zscore'

# Set to None for full-batch (original behaviour, good for ~4600 obs).
# Set to an integer e.g. 512 or 1024 for large individual-level datasets.
# The Poisson loss is mean-normalised so lr does not need adjusting.
BATCH_FRACTION = None
# Cosine annealing lr schedule — strongly recommended with minibatching
# to prevent oscillation at convergence.  Safe to keep True for full-batch too.
LR_SCHEDULE = False
USE_COMPILE = False

# ==========================================================================
# 2. DATA LOAD
# ==========================================================================
CSV_PATH = r"C:\Users\calle\OneDrive - KTH\Master's Thesis\DLNAM\chicago_nmmaps.csv"
data = pd.read_csv(CSV_PATH)

pollutant_base = ['dptp', 'o3', 'pm10']
for col in pollutant_base:
    data[f'{col}01'] = data[col].rolling(window=2).mean()

pollutant_cols = [f'{col}01' for col in pollutant_base]
data = data.dropna(
    subset=pollutant_cols + ['death', 'temp']
).reset_index(drop=True)

conf_cols = pollutant_cols

# ==========================================================================
# 3. DATA PREPARATION
# ==========================================================================
processor = DLNAMDataProcessor()
X_exposures, X_c, X_time, Y = processor.prepare_agnostic_tensors(
    df=data,
    exposure_cols=['temp'],
    lag_max_list=[30],
    conf_cols=conf_cols,
    target_col='death',
    scaling_type=EXPOSURE_SCALING,
)

# ==========================================================================
# 4. ARCHITECTURE
# ==========================================================================
surf_configs  = [[64, 64]]
conf_configs  = [[32] for _ in range(len(conf_cols))]
trend_layers  = [256, 128, 64]

# ==========================================================================
# 5. INITIALISE AND TRAIN
# ==========================================================================
trainer = Trainer(
    Multilayer_DLNAM,
    num_models=MODELS_IN_ENSEMBLE,
    exposure_dims=[30],
    conf_dim=len(conf_cols),
    surface_configs=surf_configs,
    conf_configs=conf_configs,
    num_subnets=NUM_SUBNETS,
    trend_layers=trend_layers,

    surface_activation=nn.Mish,
    # NOTE: Hardtanh(0,1) is retained for the trend network as recommended
    # by the NAM paper in the context of ExU hidden units – it acts as a
    # per-unit output clamp that prevents individual ExU units from
    # dominating the ensemble.
    trend_activation=lambda: nn.Hardtanh(0, 1),
    conf_activation=nn.Mish,

    # ExU settings
    exu_mean_val=1.0,    # Controls sharpness of exposure thresholds
    exu_mean_trend=3.5,  # Controls sharpness of seasonal wiggles
    exu_mean_lag=1.0,    # Controls sharpness of lag thresholds

    use_exu_exposure=True,
    use_exu_trend=True,
    use_exu_lag=True,

    dropout_p=0.0,
    subnet_dropout_p=0.0,   # Set e.g. 0.1 to enable NAM-style feature dropout
    output_penalty=0,
    use_compile=USE_COMPILE,
    device=device,
)

trainer.train(
    X_exposures, X_c, X_time, Y,
    epochs=EPOCHS,
    batch_fraction=BATCH_FRACTION,
    lr_schedule=LR_SCHEDULE,
    optim_kwargs={'lr': 0.0003, 'weight_decay': 0},
)

# ==========================================================================
# 6. VISUALISATION & EVALUATION
# ==========================================================================
viz = ResultVisualizer(trainer, processor)
viz.plot_all(data, exposure_cols=['temp'], conf_cols=conf_cols)

evaluator = PerformanceEvaluator(trainer)
metrics = evaluator.calculate_metrics(X_exposures, X_c, X_time, Y)
evaluator.print_report(metrics)
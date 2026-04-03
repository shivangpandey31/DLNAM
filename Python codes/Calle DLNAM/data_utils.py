# data_utils.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _make_scaler(scaling_type):
    if scaling_type == 'zscore':
        return StandardScaler()
    return MinMaxScaler(feature_range=(0, 1))


class DLNAMDataProcessor:
    def __init__(self):
        self.scalers       = {}
        self.t_means       = {}
        self.t_stds        = {}
        # Stores the scaling type for each input group so train.py can
        # read them automatically when building penalty grids.
        self.scaling_types = {}

    def prepare_agnostic_tensors(self,
                                 df,
                                 exposure_cols,
                                 lag_max_list,
                                 conf_cols,
                                 target_col,
                                 # ------------------------------------------
                                 # Per-group scaling types.
                                 # 'minmax' -> MinMaxScaler to [0, 1]
                                 # 'zscore' -> StandardScaler
                                 # Lag and trend are always [0,1] by
                                 # construction so 'minmax' is always
                                 # correct for those two — the parameters
                                 # exist for documentation and penalty grid
                                 # building only.
                                 # ------------------------------------------
                                 scaling_type_exposure='minmax',
                                 scaling_type_conf='zscore',
                                 scaling_type_trend='minmax',
                                 scaling_type_lag='minmax',
                                 # Legacy single scaling_type — overrides
                                 # exposure and conf if provided
                                 scaling_type=None,
                                 # ------------------------------------------
                                 # Categorical encodings.
                                 # List of dicts, one per categorical variable:
                                 #   {
                                 #     'name'   : str   — matches encoding_configs
                                 #     'col'    : str   — DataFrame column name
                                 #     'order'  : list  — category names in order;
                                 #                        index 0 = reference level
                                 #   }
                                 # Each category string is mapped to its integer
                                 # index in 'order'.  The resulting LongTensor is
                                 # returned in X_encodings in the same order.
                                 # ------------------------------------------
                                 encoding_configs=None):
        """
        Returns
        -------
        X_exposures  : list of FloatTensors, one per exposure
        X_conf       : FloatTensor of confounder values
        X_time       : FloatTensor of normalised time [0,1]
        Y            : FloatTensor of target counts
        X_encodings  : list of LongTensors, one per encoding (or empty list)
        """
        if scaling_type is not None:
            scaling_type_exposure = scaling_type
            scaling_type_conf     = scaling_type

        self.scaling_types['exposure'] = scaling_type_exposure
        self.scaling_types['conf']     = scaling_type_conf
        self.scaling_types['trend']    = scaling_type_trend
        self.scaling_types['lag']      = scaling_type_lag

        if encoding_configs is None:
            encoding_configs = []

        df_proc     = df.copy()
        X_exposures = []

        # ------------------------------------------------------------------
        # 1. Scale exposures
        # ------------------------------------------------------------------
        for col, lag in zip(exposure_cols, lag_max_list):
            scaler        = _make_scaler(scaling_type_exposure)
            df_proc[col]  = scaler.fit_transform(df_proc[[col]])
            self.scalers[col] = scaler
            self.t_means[col] = float(
                scaler.mean_[0] if hasattr(scaler, 'mean_') else scaler.min_[0]
            )
            self.t_stds[col]  = float(
                np.sqrt(scaler.var_[0]) if hasattr(scaler, 'var_')
                else scaler.scale_[0]
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                df_proc[col].values, lag + 1
            )
            X_exposures.append(
                torch.FloatTensor(windows.copy()).flip(dims=[1])
            )

        # ------------------------------------------------------------------
        # 2. Scale confounders
        # ------------------------------------------------------------------
        for col in conf_cols:
            scaler        = _make_scaler(scaling_type_conf)
            df_proc[col]  = scaler.fit_transform(df_proc[[col]])
            self.scalers[col] = scaler
            self.t_means[col] = float(
                scaler.mean_[0] if hasattr(scaler, 'mean_') else scaler.min_[0]
            )
            self.t_stds[col]  = float(
                np.sqrt(scaler.var_[0]) if hasattr(scaler, 'var_')
                else scaler.scale_[0]
            )

        # ------------------------------------------------------------------
        # 3. Alignment
        # ------------------------------------------------------------------
        total_lag_max = max(lag_max_list)
        X_exposures   = [
            x[total_lag_max - lag_max_list[i]:]
            for i, x in enumerate(X_exposures)
        ]

        X_conf      = torch.FloatTensor(df_proc[conf_cols].values[total_lag_max:])
        num_samples = len(X_exposures[0])

        # ------------------------------------------------------------------
        # 4. Time trend (always [0,1])
        # ------------------------------------------------------------------
        X_time = torch.FloatTensor(
            np.linspace(0, 1, num_samples)
        ).unsqueeze(1)

        # ------------------------------------------------------------------
        # 5. Target
        # ------------------------------------------------------------------
        Y = torch.tensor(
            df_proc[target_col].values[total_lag_max:].copy(),
            dtype=torch.float32
        ).unsqueeze(1)

        # ------------------------------------------------------------------
        # 6. Categorical encodings (generic — handles any number)
        # ------------------------------------------------------------------
        X_encodings = []
        for ec in encoding_configs:
            col   = ec['col']
            order = ec['order']
            enc_map = {cat: i for i, cat in enumerate(order)}
            enc_int = df[col].map(enc_map).values[total_lag_max:]
            X_encodings.append(torch.tensor(enc_int, dtype=torch.long))

        return X_exposures, X_conf, X_time, Y, X_encodings
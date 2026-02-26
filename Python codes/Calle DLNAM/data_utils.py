# data_utils.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DLNAMDataProcessor:
    def __init__(self):
        self.scalers = {}
        self.t_means = {}
        self.t_stds = {}

    def prepare_agnostic_tensors(self, df, exposure_cols, lag_max_list,
                                 conf_cols, target_col,
                                 scaling_type='minmax'):
        df_proc = df.copy()
        X_exposures = []

        # ------------------------------------------------------------------
        # 1. Scale exposures
        # ------------------------------------------------------------------
        for col, lag in zip(exposure_cols, lag_max_list):
            if scaling_type == 'zscore':
                scaler = StandardScaler()
                df_proc[col] = scaler.fit_transform(df_proc[[col]])
                self.t_means[col] = float(scaler.mean_[0])
                self.t_stds[col] = float(np.sqrt(scaler.var_[0]))
            else:
                # MinMax to [0, 1] – optimised for ExU compatibility
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_proc[col] = scaler.fit_transform(df_proc[[col]])
                self.t_means[col] = float(scaler.min_[0])
                self.t_stds[col] = float(scaler.scale_[0])

            self.scalers[col] = scaler

            windows = np.lib.stride_tricks.sliding_window_view(
                df_proc[col].values, lag + 1
            )
            # Flip so index 0 = current day, index L = furthest lag
            X_exposures.append(
                torch.FloatTensor(windows.copy()).flip(dims=[1])
            )

        # ------------------------------------------------------------------
        # 2. Scale confounders (z-score for interpretability)
        # ------------------------------------------------------------------
        for col in conf_cols:
            c_scaler = StandardScaler()
            df_proc[col] = c_scaler.fit_transform(df_proc[[col]])
            self.scalers[col] = c_scaler
            self.t_means[col] = float(c_scaler.mean_[0])
            self.t_stds[col] = float(np.sqrt(c_scaler.var_[0]))

        # ------------------------------------------------------------------
        # 3. Alignment – trim all tensors to the same length
        # ------------------------------------------------------------------
        total_lag_max = max(lag_max_list)
        X_exposures = [
            x[total_lag_max - lag_max_list[i]:]
            for i, x in enumerate(X_exposures)
        ]

        X_conf = torch.FloatTensor(
            df_proc[conf_cols].values[total_lag_max:]
        )
        num_samples = len(X_exposures[0])

        # ------------------------------------------------------------------
        # 4. Time trend (normalised 0 → 1)
        # ------------------------------------------------------------------
        X_time = torch.FloatTensor(
            np.linspace(0, 1, num_samples)
        ).unsqueeze(1)

        # ------------------------------------------------------------------
        # 5. Target variable
        # ------------------------------------------------------------------
        Y = torch.tensor(
            df_proc[target_col].values[total_lag_max:].copy(),
            dtype=torch.float32
        ).unsqueeze(1)

        return X_exposures, X_conf, X_time, Y
# visualization.py

import numpy as np
import torch
import matplotlib.pyplot as plt

from DLNAM.models import _make_lag_grid


class ResultVisualizer:
    def __init__(self, trainer, processor):
        self.trainer = trainer
        self.processor = processor

    def plot_all(self, data, exposure_cols, conf_cols):
        for idx, col_name in enumerate(exposure_cols):
            val_seq, lags, all_surfaces, ref_val = self.get_surface_data(
                data, col_name, idx
            )
            self._plot_exposure_figures(val_seq, lags, all_surfaces, ref_val, col_name)
        self.plot_trend()
        self.plot_individual_confounders(conf_cols)

    # ------------------------------------------------------------------
    # Surface data extraction
    # ------------------------------------------------------------------
    def get_surface_data(self, data, col_name, surface_idx):
        lag_max = self.trainer.exposure_lags[surface_idx]
        scaler = self.processor.scalers[col_name]

        val_min = data[col_name].min()
        val_max = data[col_name].max()
        val_seq = np.linspace(val_min, val_max, 100)
        # Number of lag steps is lag_max + 1 (lag 0 … lag_max)
        num_lags = lag_max + 1
        lags = np.arange(num_lags, dtype=float)          # original-scale lag indices

        ref_val = float(np.median(data[col_name]))

        import pandas as pd
        v_df = pd.DataFrame(val_seq.reshape(-1, 1), columns=[col_name])
        r_df = pd.DataFrame([[ref_val]], columns=[col_name])

        v_scaled_raw = scaler.transform(v_df)             # (100, 1)
        r_scaled_raw = float(scaler.transform(r_df)[0, 0])

        all_surfaces = []
        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.to('cpu').eval()

                lag_grid = _make_lag_grid(num_lags, device=torch.device('cpu'))
                # lag_grid : (num_lags,)  values in [0, 1]

                # Build the full (num_lags * 100, 2) grid in one shot.
                #
                # Layout: for each lag i we want 100 rows with the same lag
                # value paired with each of the 100 exposure values.
                #
                # v_scaled_raw : (100, 1)  – 100 exposure values
                # lag_grid     : (num_lags,)
                #
                # v_tiled  repeats the 100 exposure values num_lags times →
                #   shape (num_lags * 100, 1)
                # l_tiled  repeats each lag value 100 times →
                #   shape (num_lags * 100, 1)
                v_tiled = torch.FloatTensor(
                    np.tile(v_scaled_raw, (num_lags, 1))
                )  # (num_lags*100, 1)
                l_tiled = lag_grid.repeat_interleave(100).unsqueeze(1)
                # (num_lags*100, 1)

                r_tiled = torch.full((num_lags * 100, 1), r_scaled_raw)

                # Single forward pass for all lags and exposure values
                log_rr_flat = (
                    model.get_log_rr(v_tiled, l_tiled, surface_idx=surface_idx)
                    - model.get_log_rr(r_tiled, l_tiled, surface_idx=surface_idx)
                )  # (num_lags*100, 1)

                # Reshape to (num_lags, 100) – matches expected surface layout
                surf = log_rr_flat.squeeze(1).numpy().reshape(num_lags, 100)
                all_surfaces.append(surf)

        return val_seq, lags, np.array(all_surfaces), ref_val

    # ------------------------------------------------------------------
    # Exposure plots
    # ------------------------------------------------------------------
    def _plot_exposure_figures(self, val_seq, lags, all_surfaces, ref_val, col_name):
        mean_surf_log = np.mean(all_surfaces, axis=0)   # (L+1, 100)
        final_RR = np.exp(mean_surf_log)

        # Cumulative log-RR: sum over lag dimension for each ensemble member
        cum_log_rr_per_model = np.sum(all_surfaces, axis=1)   # (n_models, 100)
        mean_cum_log = np.mean(cum_log_rr_per_model, axis=0)  # (100,)
        sd_cum_log = np.std(cum_log_rr_per_model, axis=0)     # (100,)

        # 1. Cumulative Effect
        plt.figure(figsize=(8, 5))
        plt.plot(val_seq, np.exp(mean_cum_log),
                 color='firebrick', lw=2, label='Ensemble Mean')
        plt.fill_between(
            val_seq,
            np.exp(mean_cum_log - 1.96 * sd_cum_log),
            np.exp(mean_cum_log + 1.96 * sd_cum_log),
            color='firebrick', alpha=0.15, label='Ensemble ±1.96 SD'
        )
        plt.axhline(1, color='black', ls='--', lw=1)
        plt.axvline(ref_val, color='gray', ls=':', label='Median')
        plt.title(f"Cumulative Effect: {col_name}")
        plt.xlabel(f"{col_name} Value")
        plt.ylabel("Relative Risk (RR)")
        plt.legend(frameon=True)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        # 2. 3-D Surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        V, L = np.meshgrid(val_seq, lags)
        surf_3d = ax.plot_surface(
            V, L, final_RR, cmap='RdYlBu_r', edgecolor='none', alpha=0.9
        )
        ax.set_xlabel(col_name)
        ax.set_ylabel("Lag (Days)")
        ax.set_zlabel("RR")
        ax.set_title(f"3D Exposure-Lag-Response Surface: {col_name}")
        fig.colorbar(surf_3d, ax=ax, shrink=0.5, aspect=10, label='RR')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Trend plot
    # ------------------------------------------------------------------
    def plot_trend(self):
        time_seq = torch.linspace(0, 1, 500).view(-1, 1)
        all_trends = []

        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.to('cpu').eval()
                trend_eff = torch.zeros((500, 1))
                w_ens_t = torch.softmax(model.trend_weights[0], dim=0)
                for s, g_net in enumerate(model.trend_subnets):
                    trend_eff = trend_eff + w_ens_t[s] * g_net(time_seq)
                all_trends.append(trend_eff.numpy())

        # Post-hoc mean-centering so the trend is interpretable as a
        # deviation from the temporal average
        all_trends_array = np.array(all_trends)           # (n_models, 500, 1)
        centered_trends = [t - np.mean(t) for t in all_trends_array]
        mean_t = np.mean(centered_trends, axis=0)         # (500, 1)
        std_t = np.std(centered_trends, axis=0)

        plt.figure(figsize=(10, 4))
        plt.plot(time_seq.numpy(), mean_t,
                 color='teal', lw=2, label='Ensemble Mean Trend')
        plt.fill_between(
            time_seq.numpy().flatten(),
            (mean_t - 1.96 * std_t).flatten(),
            (mean_t + 1.96 * std_t).flatten(),
            color='teal', alpha=0.15, label='Ensemble ±1.96 SD'
        )
        plt.axhline(0, color='black', ls='--', lw=0.8)
        plt.title("Long-term Trend / Seasonality")
        plt.xlabel("Time (Normalised 0–1)")
        plt.ylabel("Log-Relative Effect")
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Confounder plots
    # ------------------------------------------------------------------
    def plot_individual_confounders(self, conf_names):
        # Evaluate over ±3 SD of the z-score-scaled confounders
        eval_range = torch.linspace(-3, 3, 100).view(-1, 1)
        covariate_color = '#455a64'

        for k, name in enumerate(conf_names):
            all_eff = []
            with torch.no_grad():
                for model in self.trainer.ensemble:
                    model.to('cpu').eval()
                    eff = torch.zeros((100, 1))
                    w_ens = torch.softmax(model.conf_weights[k], dim=0)
                    for s, subnet in enumerate(model.conf_subnets[k]):
                        eff = eff + w_ens[s] * subnet(eval_range)
                    all_eff.append(eff.numpy())

            # Centre at the mean (index 50 ≈ z-score 0)
            all_eff_array = np.array(all_eff)
            centered_conf = [e - e[50] for e in all_eff_array]
            m = np.mean(centered_conf, axis=0)
            s = np.std(centered_conf, axis=0)

            plt.figure(figsize=(7, 4))
            plt.plot(eval_range.numpy(), m,
                     color=covariate_color, lw=2, label='Ensemble Mean')
            plt.fill_between(
                eval_range.numpy().flatten(),
                (m - 1.96 * s).flatten(),
                (m + 1.96 * s).flatten(),
                color=covariate_color, alpha=0.15, label='Ensemble ±1.96 SD'
            )
            plt.axhline(0, color='black', ls='--', lw=0.8)
            plt.axvline(0, color='gray', ls=':', lw=0.8, label='Mean Exposure')
            plt.title(f"Effect of {name}")
            plt.xlabel("Exposure Concentration (Z-Score)")
            plt.ylabel("Log-Relative Effect")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
# visualization.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from DLNAM.models import _make_lag_grid


# ------------------------------------------------------------------
# CI helper
# ------------------------------------------------------------------
def _compute_ci(mean_arr, sd_arr, alpha, ci_type, phi=1.0):
    z   = stats.norm.ppf(1.0 - alpha / 2.0)
    pct = int(round((1.0 - alpha) * 100))

    if ci_type == 'ensemble':
        lo    = mean_arr - z * sd_arr
        hi    = mean_arr + z * sd_arr
        label = f"Ensemble {pct}% CI"
    elif ci_type == 'poisson':
        mu    = np.clip(mean_arr, 1e-8, None)
        lo    = stats.poisson.ppf(alpha / 2.0,       mu=mu)
        hi    = stats.poisson.ppf(1.0 - alpha / 2.0, mu=mu)
        label = f"Poisson {pct}% CI"
    elif ci_type == 'wald':
        mu    = np.clip(mean_arr, 1e-8, None)
        se    = np.sqrt(phi * mu)
        lo    = mean_arr - z * se
        hi    = mean_arr + z * se
        label = f"Wald {pct}% CI"
    else:
        raise ValueError(f"ci_type must be 'ensemble', 'poisson', or 'wald'. Got: {ci_type}")

    return lo, hi, label


class ResultVisualizer:
    def __init__(self, trainer, processor,
                 alpha=0.05,
                 ci_type='ensemble',
                 centering='median',
                 phi=None):
        self.trainer   = trainer
        self.processor = processor
        self.alpha     = alpha
        self.ci_type   = ci_type
        self.centering = centering
        self.phi       = phi

    def plot_all(self, data, exposure_cols, conf_cols,
                 Y=None, X_exposures=None, X_c=None, X_time=None,
                 X_encodings=None, encoding_configs=None):
        if self.ci_type == 'wald' and self.phi is None:
            if Y is None:
                raise ValueError(
                    "Pass Y and X_* to plot_all() so phi can be estimated, "
                    "or supply phi= at construction."
                )
            self.phi = self._estimate_phi(X_exposures, X_c, X_time, Y)
            print(f"  Estimated phi for Wald intervals: {self.phi:.4f}")

        if self.centering == 'mean' and X_exposures is None:
            raise ValueError(
                "Pass X_exposures, X_c, X_time to plot_all() when centering='mean'."
            )

        for idx, col_name in enumerate(exposure_cols):
            val_seq, lags, cum_surfaces, per_lag_surfaces, ref_idx = \
                self.get_surface_data(data, col_name, idx, X_exposures=X_exposures)
            self._plot_exposure_figures(
                val_seq, lags, cum_surfaces, per_lag_surfaces, ref_idx, col_name
            )

        self.plot_trend(X_time=X_time)
        self.plot_individual_confounders(conf_cols, X_c=X_c)

        if X_encodings:
            self.plot_encodings(encoding_configs=encoding_configs)

        self.plot_training_loss()

    def plot_training_loss(self, show_ci=True, label=None, ax=None):
        """
        Plots the ensemble mean training loss vs epoch with an optional
        shaded band showing the ensemble 95% CI across members.

        Parameters
        ----------
        show_ci : bool
            If True (default), draw the shaded ensemble CI band.
            Set to False for benchmark overlays where only the mean is needed.
        label   : str or None
            Line label for the legend. Defaults to 'Ensemble Mean'.
        ax      : matplotlib Axes or None
            If provided, plot into this axes (for benchmark overlays).
            If None, creates a new figure.
        """
        if not hasattr(self.trainer, 'loss_history') or not self.trainer.loss_history:
            print("No loss history available — train the model first.")
            return

        ORANGE      = '#FF8C00'
        ORANGE_FILL = '#FFAD60'

        histories = self.trainer.loss_history
        epochs    = [h[0] for h in histories[0]]
        losses    = np.array([[h[1] for h in hist] for hist in histories])
        # losses: (n_members, n_checkpoints)

        mean_loss = np.mean(losses, axis=0)
        std_loss  = np.std(losses,  axis=0)
        z         = stats.norm.ppf(1.0 - self.alpha / 2.0)
        pct       = int(round((1.0 - self.alpha) * 100))
        lo        = mean_loss - z * std_loss
        hi        = mean_loss + z * std_loss

        # Skip the first 5% of epochs to remove the initialisation spike
        warmup    = max(1, int(0.05 * len(epochs)))
        epochs    = epochs[warmup:]
        mean_loss = mean_loss[warmup:]
        lo        = lo[warmup:]
        hi        = hi[warmup:]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 4))

        line_label = label if label is not None else 'Ensemble Mean'
        ax.plot(epochs, mean_loss, color=ORANGE, lw=2.0, label=line_label)

        if show_ci:
            ax.fill_between(epochs, lo, hi,
                            color=ORANGE_FILL, alpha=0.35,
                            label=f'Ensemble {pct}% CI')

        if standalone:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Poisson Loss')
            ax.set_title('Training Loss')
            ax.set_xlim(epochs[0], epochs[-1])
            ax.legend(frameon=True, fontsize=8)
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

    def plot_encodings(self, encoding_configs=None):
        """
        Dot plot of learned categorical encoding effects for every encoding
        in the model, expressed as RR relative to the first category (index 0).
        One plot per encoding.
        """
        if not any(len(m.encodings) > 0 for m in self.trainer.ensemble):
            return

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        n_enc = len(self.trainer.ensemble[0].encodings)

        for enc_idx in range(n_enc):
            # Collect weights across ensemble
            all_weights = []
            for model in self.trainer.ensemble:
                model.to('cpu').eval()
                # The input layer weight matrix has shape (1, C) or
                # (hidden_dim, C); row 0 gives the per-category scalars
                # when hidden_layers=[] (pure lookup table).
                # For deeper networks we extract the full forward pass
                # at each one-hot basis vector instead.
                enc_mod = model.encodings[enc_idx]
                n_cats_local = enc_mod.num_categories
                eye = torch.eye(n_cats_local)
                with torch.no_grad():
                    w = enc_mod.net(eye).squeeze(1).numpy()
                all_weights.append(w)
            for m in self.trainer.ensemble:
                m.to(self.trainer.device)

            all_weights = np.array(all_weights)   # (N_ensemble, C)
            n_cats      = all_weights.shape[1]   # == num_categories

            # Determine labels and name from encoding_configs if provided
            if encoding_configs is not None and enc_idx < len(encoding_configs):
                ec     = encoding_configs[enc_idx]
                name   = ec.get('name', f'Encoding {enc_idx}')
                labels = ec.get('order', [str(i) for i in range(n_cats)])
            else:
                name   = f'Encoding {enc_idx}'
                labels = [str(i) for i in range(n_cats)]

            # Centre at the median category index so RR=1 at the middle
            # category — e.g. Thursday (index 3) for a Mon-Sun encoding.
            # This matches the DLNM cenvalue convention.
            ref_idx  = n_cats // 2
            all_w_c  = all_weights - all_weights[:, ref_idx:ref_idx + 1]
            mean_w_c = np.mean(all_w_c, axis=0)
            sd_w_c   = np.std(all_w_c,  axis=0)

            mean_rr = np.exp(mean_w_c)
            sd_rr   = mean_rr * sd_w_c   # delta method

            z   = stats.norm.ppf(1.0 - self.alpha / 2.0)
            lo  = mean_rr - z * sd_rr
            hi  = mean_rr + z * sd_rr
            pct = int(round((1.0 - self.alpha) * 100))

            x         = np.arange(n_cats)
            DOT_COLOR = '#0a1628'

            fig, ax = plt.subplots(figsize=(max(6, n_cats), 4))

            for i in range(n_cats):
                ax.plot([x[i], x[i]], [lo[i], hi[i]],
                        color=DOT_COLOR, lw=1.2, zorder=2, alpha=0.6)
            for i in range(n_cats):
                ax.scatter(x[i], mean_rr[i], color=DOT_COLOR, s=60, zorder=3,
                           edgecolors='white', linewidths=0.5)

            legend_elements = [
                Line2D([0], [0], color=DOT_COLOR, lw=2, label='Ensemble Mean'),
                Patch(facecolor=DOT_COLOR, alpha=0.4,
                      label=f'Ensemble {pct}% CI'),
            ]
            ax.axhline(1, color='black', lw=1.2, ls='-', zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha='center')
            ax.set_ylabel('Relative Risk (RR)')
            ax.set_title(f'Encoding Effect: {name}')
            ax.legend(handles=legend_elements, frameon=True, fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _estimate_phi(self, X_exposures, X_c, X_time, Y):
        device   = self.trainer.device
        X_exp_d  = [x.to(device) for x in X_exposures]
        X_c_d    = X_c.to(device)
        X_time_d = X_time.to(device)
        Y_np     = Y.cpu().numpy().flatten()
        all_preds = []
        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.eval()
                p = model(X_exp_d, X_c_d, X_time_d).cpu().numpy().flatten()
                all_preds.append(p)
        mu_hat = np.mean(all_preds, axis=0)
        return float(np.mean((Y_np - mu_hat) ** 2 / (mu_hat + 1e-8)))

    def _anchor(self, log_effects_grid, log_effects_train=None, ref_idx=None):
        if self.centering == 'median':
            return float(log_effects_grid[ref_idx])
        else:
            return float(np.mean(log_effects_train))

    def _train_surface_log(self, model, surface_idx, X_exposures):
        device    = next(model.parameters()).device
        x_m       = X_exposures[surface_idx].to(device)
        B         = x_m.shape[0]
        curr_lags = x_m.shape[1]
        lag_grid  = _make_lag_grid(curr_lags, device)
        v_exp     = x_m.reshape(-1, 1)
        l_exp     = lag_grid.repeat(B).unsqueeze(1)
        surf_input = torch.cat([v_exp, l_exp], dim=1)
        w_ens     = model._mix_weights(model.exp_weights[surface_idx])
        stacked   = torch.stack(
            [s(surf_input) for s in model.surface_subnets[surface_idx]], dim=0
        )
        feat_flat = (w_ens.view(-1, 1, 1) * stacked).sum(dim=0)
        return feat_flat.view(B, curr_lags).sum(dim=1).cpu().numpy()

    def _train_conf_log(self, model, conf_idx, X_c):
        device  = next(model.parameters()).device
        val_k   = X_c[:, conf_idx:conf_idx + 1].to(device)
        w_ens   = model._mix_weights(model.conf_weights[conf_idx])
        stacked = torch.stack(
            [s(val_k) for s in model.conf_subnets[conf_idx]], dim=0
        )
        return (w_ens.view(-1, 1, 1) * stacked).sum(dim=0).squeeze(1).cpu().numpy()

    def _train_trend_log(self, model, X_time):
        device   = next(model.parameters()).device
        x_time_d = X_time.to(device)
        w_ens    = model._mix_weights(model.trend_weights[0])
        stacked  = torch.stack(
            [s(x_time_d) for s in model.trend_subnets], dim=0
        )
        return (w_ens.view(-1, 1, 1) * stacked).sum(dim=0).squeeze(1).cpu().numpy()

    def get_surface_data(self, data, col_name, surface_idx, X_exposures=None):
        import pandas as pd

        lag_max  = self.trainer.exposure_lags[surface_idx]
        scaler   = self.processor.scalers[col_name]

        val_min  = data[col_name].min()
        val_max  = data[col_name].max()
        val_seq  = np.linspace(val_min, val_max, 100)
        num_lags = lag_max + 1
        lags     = np.arange(num_lags, dtype=float)

        ref_val  = float(np.median(data[col_name]))
        ref_idx  = int(np.argmin(np.abs(val_seq - ref_val)))

        v_df         = pd.DataFrame(val_seq.reshape(-1, 1), columns=[col_name])
        r_df         = pd.DataFrame([[ref_val]], columns=[col_name])
        v_scaled_raw = scaler.transform(v_df)
        r_scaled_raw = float(scaler.transform(r_df)[0, 0])

        all_surfaces = []
        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.to('cpu').eval()

                lag_grid = _make_lag_grid(num_lags, device=torch.device('cpu'))
                v_tiled  = torch.FloatTensor(np.tile(v_scaled_raw, (num_lags, 1)))
                l_tiled  = lag_grid.repeat_interleave(100).unsqueeze(1)
                r_tiled  = torch.full((num_lags * 100, 1), r_scaled_raw)

                log_per_lag = (
                    model.get_log_rr(v_tiled, l_tiled, surface_idx=surface_idx)
                    - model.get_log_rr(r_tiled, l_tiled, surface_idx=surface_idx)
                ).squeeze(1).numpy().reshape(num_lags, 100)

                cum_log = log_per_lag.sum(axis=0)

                if self.centering == 'mean':
                    train_log = self._train_surface_log(
                        model, surface_idx, X_exposures
                    )
                    anchor = self._anchor(cum_log, log_effects_train=train_log)
                    all_surfaces.append({
                        'cum':     cum_log - anchor,
                        'per_lag': log_per_lag - anchor / num_lags,
                    })
                else:
                    all_surfaces.append({
                        'cum':     cum_log,
                        'per_lag': log_per_lag,
                    })

        for m in self.trainer.ensemble:
            m.to(self.trainer.device)

        cum_surfaces     = np.array([s['cum']     for s in all_surfaces])
        per_lag_surfaces = np.array([s['per_lag'] for s in all_surfaces])
        return val_seq, lags, cum_surfaces, per_lag_surfaces, ref_idx

    def _plot_exposure_figures(self, val_seq, lags, cum_surfaces,
                               per_lag_surfaces, ref_idx, col_name):
        mean_cum_log = np.mean(cum_surfaces, axis=0)
        sd_cum_log   = np.std(cum_surfaces, axis=0)
        mean_rr      = np.exp(mean_cum_log)
        sd_rr        = mean_rr * sd_cum_log
        ref_val      = val_seq[ref_idx]

        lo, hi, ci_label = _compute_ci(
            mean_rr, sd_rr,
            alpha=self.alpha, ci_type=self.ci_type,
            phi=self.phi if self.phi is not None else 1.0,
        )

        # 1. Cumulative effect
        plt.figure(figsize=(8, 5))
        plt.plot(val_seq, mean_rr, color='firebrick', lw=2, label='Ensemble Mean')
        plt.fill_between(val_seq, lo, hi,
                         color='firebrick', alpha=0.15, label=ci_label)
        plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
        plt.axvline(ref_val, color='gray', ls=':', lw=0.8, label='_nolegend_')
        plt.xlim(val_seq[0], val_seq[-1])
        plt.title(f"Cumulative Effect: {col_name}")
        plt.xlabel(f"{col_name} Value")
        plt.ylabel("Relative Risk (RR)")
        plt.legend(frameon=True)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # Shared colour setup for 3D and contour
        # ------------------------------------------------------------------
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        import matplotlib.cm as cm

        mean_per_lag_log = np.mean(per_lag_surfaces, axis=0)  # (L+1, 100)
        mean_per_lag_rr  = np.exp(mean_per_lag_log)

        vmin = min(mean_per_lag_rr.min(), 0.999)
        vmax = max(mean_per_lag_rr.max(), 1.001)

        # TwoSlopeNorm centres white exactly at RR=1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

        # Custom colormap matching other plots
        cmap = LinearSegmentedColormap.from_list(
            'dlnam', ['#0a1628', '#ffffff', '#B22222'], N=512
        )

        # Equidistant colorbar ticks — same for both plots
        n_ticks  = 8
        cb_ticks = np.linspace(vmin, vmax, n_ticks)

        V, L = np.meshgrid(val_seq, lags)

        # face_colors maps each RR value through norm+cmap for a smooth
        # continuous gradient identical to the contour fill.
        face_colors = cmap(norm(mean_per_lag_rr))  # (L+1, 100, 4) RGBA

        # 2. 3D surface — rcount/ccount match grid resolution to eliminate
        #    the net/grid appearance in favour of a smooth gradient
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            V, L, mean_per_lag_rr,
            facecolors=face_colors,
            edgecolor='none',
            alpha=0.95,
            shade=False,
            rcount=200,
            ccount=200,
            antialiased=False,
        )

        # Colorbar driven by a ScalarMappable with equidistant ticks
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10,
                          label='RR', ticks=cb_ticks)
        cb.ax.set_yticklabels([f'{t:.2f}' for t in cb_ticks])

        # White panes with subtle grey edges — clean publication style
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor('white')
            pane.set_edgecolor('#cccccc')
            pane.set_linewidth(0.5)
        ax.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', alpha=0.7)

        ax.set_xlabel(col_name, labelpad=8)
        ax.set_ylabel('Lag (Days)', labelpad=8)
        ax.set_zlabel('RR', labelpad=8)
        ax.set_title(f'3D Exposure-Lag-Response: {col_name}')
        plt.tight_layout()
        plt.show()

        # 3. Contour plot — identical norm, cmap and colorbar ticks as 3D
        fig, ax = plt.subplots(figsize=(8, 5))
        cntr    = ax.contourf(V, L, mean_per_lag_rr,
                              levels=200, cmap=cmap, norm=norm)
        cbar_c  = fig.colorbar(cntr, ax=ax, label='RR', ticks=cb_ticks)
        cbar_c.ax.set_yticklabels([f'{t:.2f}' for t in cb_ticks])
        ax.set_xlim(val_seq[0], val_seq[-1])
        ax.set_ylim(lags[0], lags[-1])
        ax.set_xlabel(col_name)
        ax.set_ylabel('Lag (Days)')
        ax.set_title(f'Exposure-Lag-Response Contour: {col_name}')
        plt.tight_layout()
        plt.show()

    def plot_trend(self, X_time=None):
        time_seq = torch.linspace(0, 1, 500).view(-1, 1)
        ref_idx  = 250
        all_trends = []

        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.to('cpu').eval()
                w_ens_t = model._mix_weights(model.trend_weights[0])
                stacked = torch.stack(
                    [g(time_seq) for g in model.trend_subnets], dim=0
                )
                log_raw = (
                    w_ens_t.view(-1, 1, 1) * stacked
                ).sum(dim=0).numpy().flatten()

                if self.centering == 'mean' and X_time is not None:
                    train_log = self._train_trend_log(model, X_time)
                    anchor    = self._anchor(log_raw, log_effects_train=train_log)
                else:
                    anchor = self._anchor(log_raw, ref_idx=ref_idx)

                all_trends.append(log_raw - anchor)

        for m in self.trainer.ensemble:
            m.to(self.trainer.device)

        all_trends = np.array(all_trends)
        mean_log   = np.mean(all_trends, axis=0)
        sd_log     = np.std(all_trends, axis=0)
        mean_rr    = np.exp(mean_log)
        sd_rr      = mean_rr * sd_log

        lo, hi, ci_label = _compute_ci(
            mean_rr, sd_rr,
            alpha=self.alpha, ci_type=self.ci_type,
            phi=self.phi if self.phi is not None else 1.0,
        )

        plt.figure(figsize=(10, 4))
        plt.plot(time_seq.numpy().flatten(), mean_rr,
                 color='teal', lw=2, label='Ensemble Mean')
        plt.fill_between(time_seq.numpy().flatten(), lo, hi,
                         color='teal', alpha=0.15, label=ci_label)
        plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
        plt.axvline(0.5, color='gray', ls=':', lw=0.8, label='_nolegend_')
        plt.xlim(0.0, 1.0)
        plt.title("Long-term Trend / Seasonality")
        plt.xlabel("Time (Normalised 0–1)")
        plt.ylabel("Relative Risk (RR)")
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_individual_confounders(self, conf_names, X_c=None):
        eval_range      = torch.linspace(-3, 3, 100).view(-1, 1)
        covariate_color = '#455a64'
        ref_idx         = 50

        for k, name in enumerate(conf_names):
            all_eff = []
            with torch.no_grad():
                for model in self.trainer.ensemble:
                    model.to('cpu').eval()
                    w_ens   = model._mix_weights(model.conf_weights[k])
                    stacked = torch.stack(
                        [s(eval_range) for s in model.conf_subnets[k]], dim=0
                    )
                    log_raw = (
                        w_ens.view(-1, 1, 1) * stacked
                    ).sum(dim=0).numpy().flatten()

                    if self.centering == 'mean' and X_c is not None:
                        train_log = self._train_conf_log(model, k, X_c)
                        anchor    = self._anchor(log_raw, log_effects_train=train_log)
                    else:
                        anchor = self._anchor(log_raw, ref_idx=ref_idx)

                    all_eff.append(log_raw - anchor)

            all_eff  = np.array(all_eff)
            mean_log = np.mean(all_eff, axis=0)
            sd_log   = np.std(all_eff, axis=0)
            mean_rr  = np.exp(mean_log)
            sd_rr    = mean_rr * sd_log

            lo, hi, ci_label = _compute_ci(
                mean_rr, sd_rr,
                alpha=self.alpha, ci_type=self.ci_type,
                phi=self.phi if self.phi is not None else 1.0,
            )

            plt.figure(figsize=(7, 4))
            plt.plot(eval_range.numpy().flatten(), mean_rr,
                     color=covariate_color, lw=2, label='Ensemble Mean')
            plt.fill_between(eval_range.numpy().flatten(), lo, hi,
                             color=covariate_color, alpha=0.15, label=ci_label)
            plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
            plt.axvline(0, color='gray', ls=':', lw=0.8, label='_nolegend_')
            plt.xlim(-3, 3)
            plt.title(f"Effect of {name}")
            plt.xlabel("Exposure Concentration (Z-Score)")
            plt.ylabel("Relative Risk (RR)")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

        for m in self.trainer.ensemble:
            m.to(self.trainer.device)
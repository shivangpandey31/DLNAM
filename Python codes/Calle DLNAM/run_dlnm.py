# run_dlnm.py
#
# Runs the DLNM baseline on Chicago NMMAPS by calling Rscript.exe as a
# subprocess, writing results to CSV, then reading them back into Python.
# This avoids all rpy2 embedding issues on Windows.
#
# Evaluates and visualises using the same functions as run_dlnam.py.
#
# Structured around three functions:
#   fit_dlnm()   — writes an R script, runs it, reads back CSVs
#   main()       — configuration and entry point
#
# Requires: R with packages dlnm, splines installed.

import os
import sys
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

from DLNAM.evaluation    import PerformanceEvaluator
from DLNAM.visualization import ResultVisualizer, _compute_ci
from run_dlnam           import load_chicago


# ==========================================================================
# MOCK TRAINER
# Wraps DLNM predictions in an object that satisfies the interfaces
# expected by PerformanceEvaluator and ResultVisualizer.
# ==========================================================================

class DLNMTrainer:
    """
    Lightweight wrapper around fitted DLNM predictions that mimics the
    Trainer interface used by PerformanceEvaluator and ResultVisualizer.

    Parameters
    ----------
    mu_hat        : (n,) array of fitted values from the DLNM
    cum_rr        : (n_grid, L+1) array of per-lag log-RR on prediction grid
    cum_log_rr    : (n_grid,) array of cumulative log-RR on prediction grid
    val_seq       : (n_grid,) array of exposure values for the prediction grid
    lags          : (L+1,) array of lag indices
    ref_idx       : index of the reference (median) exposure in val_seq
    trend_rr      : (500,) array of trend log-RR on normalised time grid
    conf_rr       : dict mapping confounder name -> (100,) log-RR array
    phi           : estimated dispersion parameter
    exposure_lags : list of max lags per exposure, e.g. [30]
    device        : torch device (for interface compatibility)
    """
    def __init__(self, mu_hat, cum_rr, cum_log_rr, cum_lo, cum_hi,
                 per_lag_lo, per_lag_hi,
                 val_seq, lags, ref_idx,
                 trend_rr, conf_rr, dow_rr, phi, exposure_lags, device):
        self.mu_hat        = mu_hat
        self.cum_rr        = cum_rr
        self.cum_log_rr    = cum_log_rr
        self.cum_lo        = cum_lo
        self.cum_hi        = cum_hi
        self.per_lag_lo    = per_lag_lo
        self.per_lag_hi    = per_lag_hi
        self.val_seq       = val_seq
        self.lags          = lags
        self.ref_idx       = ref_idx
        self.trend_rr      = trend_rr
        self.conf_rr       = conf_rr
        self.dow_rr        = dow_rr
        self.phi           = phi
        self.exposure_lags = exposure_lags
        self.device        = device

        # PerformanceEvaluator iterates over self.ensemble — provide a single
        # mock member whose forward() returns mu_hat as a tensor
        mu_tensor = torch.FloatTensor(mu_hat.copy()).unsqueeze(1)

        class _MockModel:
            def __init__(self, mu):
                self._mu = mu
            def eval(self): pass
            def __call__(self, *args, **kwargs):
                return self._mu

        self.ensemble = [_MockModel(mu_tensor)]


# ==========================================================================
# DLNM VISUALISER
# Subclasses ResultVisualizer, overriding the plot methods to use the
# pre-computed DLNM arrays stored in DLNMTrainer instead of running
# neural network forward passes.
# ==========================================================================

class DLNMVisualizer(ResultVisualizer):
    """
    Uses the same plot functions as ResultVisualizer but draws data
    from the pre-computed DLNMTrainer arrays rather than running the
    neural network ensemble.
    """

    def plot_all(self, data, exposure_cols, conf_cols,
                 Y=None, X_exposures=None, X_c=None, X_time=None,
                 X_encodings=None, encoding_configs=None):
        """Override to always include the DoW plot when available."""
        for idx, col_name in enumerate(exposure_cols):
            val_seq, lags, cum_surfaces, per_lag_surfaces, ref_idx = \
                self.get_surface_data(data, col_name, idx)
            self._plot_exposure_figures(
                val_seq, lags, cum_surfaces, per_lag_surfaces, ref_idx, col_name
            )
        self.plot_trend()
        self.plot_individual_confounders(conf_cols)
        if hasattr(self.trainer, 'dow_rr') and self.trainer.dow_rr is not None:
            self.plot_encodings()
        self.plot_training_loss()

    def get_surface_data(self, data, col_name, surface_idx, X_exposures=None):
        t = self.trainer
        cum_log      = t.cum_log_rr
        per_lag_log  = np.log(np.maximum(t.cum_rr, 1e-8))
        cum_surfaces     = cum_log[np.newaxis, :]
        per_lag_surfaces = per_lag_log[np.newaxis, :, :]
        return t.val_seq, t.lags, cum_surfaces, per_lag_surfaces, t.ref_idx

    def _plot_exposure_figures(self, val_seq, lags, cum_surfaces,
                               per_lag_surfaces, ref_idx, col_name):
        """Override to use R analytical CIs instead of ensemble spread."""
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        import matplotlib.cm as cm

        t       = self.trainer
        pct     = int(round((1.0 - self.alpha) * 100))
        ref_val = val_seq[ref_idx]

        # Cumulative effect
        mean_rr = np.exp(t.cum_log_rr)
        lo_rr   = np.exp(t.cum_lo)
        hi_rr   = np.exp(t.cum_hi)

        plt.figure(figsize=(8, 5))
        plt.plot(val_seq, mean_rr, color='firebrick', lw=2, label='DLNM Fit')
        plt.fill_between(val_seq, lo_rr, hi_rr,
                         color='firebrick', alpha=0.15,
                         label=f'DLNM {pct}% CI')
        plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
        plt.axvline(ref_val, color='gray', ls=':', lw=0.8, label='_nolegend_')
        plt.xlim(val_seq[0], val_seq[-1])
        plt.title(f'Cumulative Effect: {col_name} (DLNM)')
        plt.xlabel(f'{col_name} Value')
        plt.ylabel('Relative Risk (RR)')
        plt.legend(frameon=True)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        # Shared colormap
        mean_per_lag_rr = t.cum_rr                         # (L+1, n_grid)
        vmin = min(mean_per_lag_rr.min(), 0.999)
        vmax = max(mean_per_lag_rr.max(), 1.001)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            'dlnam', ['#0a1628', '#ffffff', '#B22222'], N=512)
        n_ticks  = 8
        cb_ticks = np.linspace(vmin, vmax, n_ticks)
        V, L     = np.meshgrid(val_seq, lags)
        face_colors = cmap(norm(mean_per_lag_rr))

        # 3D surface
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(V, L, mean_per_lag_rr, facecolors=face_colors,
                        edgecolor='none', alpha=0.95, shade=False,
                        rcount=200, ccount=200, antialiased=False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10,
                          label='RR', ticks=cb_ticks)
        cb.ax.set_yticklabels([f'{t2:.2f}' for t2 in cb_ticks])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor('white')
            pane.set_edgecolor('#cccccc')
            pane.set_linewidth(0.5)
        ax.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', alpha=0.7)
        ax.set_xlabel(col_name, labelpad=8)
        ax.set_ylabel('Lag (Days)', labelpad=8)
        ax.set_zlabel('RR', labelpad=8)
        ax.set_title(f'3D Exposure-Lag-Response: {col_name} (DLNM)')
        plt.tight_layout()
        plt.show()

        # Contour
        fig, ax = plt.subplots(figsize=(8, 5))
        cntr   = ax.contourf(V, L, mean_per_lag_rr,
                             levels=200, cmap=cmap, norm=norm)
        cbar_c = fig.colorbar(cntr, ax=ax, label='RR', ticks=cb_ticks)
        cbar_c.ax.set_yticklabels([f'{t2:.2f}' for t2 in cb_ticks])
        ax.set_xlim(val_seq[0], val_seq[-1])
        ax.set_ylim(lags[0], lags[-1])
        ax.set_xlabel(col_name)
        ax.set_ylabel('Lag (Days)')
        ax.set_title(f'Exposure-Lag-Response Contour: {col_name} (DLNM)')
        plt.tight_layout()
        plt.show()

    def plot_trend(self, X_time=None):
        t        = self.trainer
        time_seq = np.linspace(0, 1, len(t.trend_rr['mean']))
        pct      = int(round((1.0 - self.alpha) * 100))
        z        = stats.norm.ppf(1.0 - self.alpha / 2.0)

        log_mean = t.trend_rr['mean']
        log_se   = t.trend_rr['se']
        mean_rr  = np.exp(log_mean)
        lo_rr    = np.exp(log_mean - z * log_se)
        hi_rr    = np.exp(log_mean + z * log_se)

        plt.figure(figsize=(10, 4))
        plt.plot(time_seq, mean_rr, color='teal', lw=2, label='DLNM Fit')
        plt.fill_between(time_seq, lo_rr, hi_rr,
                         color='teal', alpha=0.15, label=f'DLNM {pct}% CI')
        plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
        plt.axvline(0.5, color='gray', ls=':', lw=0.8, label='_nolegend_')
        plt.xlim(0.0, 1.0)
        plt.title('Long-term Trend / Seasonality (DLNM)')
        plt.xlabel('Time (Normalised 0–1)')
        plt.ylabel('Relative Risk (RR)')
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_individual_confounders(self, conf_names, X_c=None):
        t               = self.trainer
        covariate_color = '#455a64'
        eval_range      = np.linspace(-3, 3, 100)
        pct             = int(round((1.0 - self.alpha) * 100))
        z               = stats.norm.ppf(1.0 - self.alpha / 2.0)

        for name in conf_names:
            if name not in t.conf_rr:
                continue
            log_mean = t.conf_rr[name]['mean']
            log_se   = t.conf_rr[name]['se']
            mean_rr  = np.exp(log_mean)
            lo_rr    = np.exp(log_mean - z * log_se)
            hi_rr    = np.exp(log_mean + z * log_se)

            plt.figure(figsize=(7, 4))
            plt.plot(eval_range, mean_rr,
                     color=covariate_color, lw=2, label='DLNM Fit')
            plt.fill_between(eval_range, lo_rr, hi_rr,
                             color=covariate_color, alpha=0.15,
                             label=f'DLNM {pct}% CI')
            plt.axhline(1, color='black', ls='-', lw=1.4, label='_nolegend_')
            plt.axvline(0, color='gray', ls=':', lw=0.8, label='_nolegend_')
            plt.xlim(-3, 3)
            plt.title(f'Effect of {name} (DLNM)')
            plt.xlabel('Exposure Concentration (Z-Score)')
            plt.ylabel('Relative Risk (RR)')
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()

    def plot_training_loss(self, **kwargs):
        pass

    def plot_encodings(self, encoding_configs=None, **kwargs):
        """
        Dot plot of DLNM day-of-week effects using the GLM coefficients
        extracted from the R model. Matches the style of the DLNAM dow plot.
        """
        t   = self.trainer
        pct = int(round((1.0 - self.alpha) * 100))
        z   = stats.norm.ppf(1.0 - self.alpha / 2.0)

        dow_labels = list(t.dow_rr['day'])
        log_mean   = t.dow_rr['mean']
        log_se     = t.dow_rr['se']
        mean_rr    = np.exp(log_mean)
        lo_rr      = np.exp(log_mean - z * log_se)
        hi_rr      = np.exp(log_mean + z * log_se)

        x         = np.arange(7)
        DOT_COLOR = '#0a1628'

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Line2D([0], [0], color=DOT_COLOR, lw=2,  label='DLNM Fit'),
            Patch(facecolor=DOT_COLOR, alpha=0.4,
                  label=f'DLNM {pct}% CI'),
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(7):
            ax.plot([x[i], x[i]], [lo_rr[i], hi_rr[i]],
                    color=DOT_COLOR, lw=1.2, zorder=2, alpha=0.6)
        for i in range(7):
            ax.scatter(x[i], mean_rr[i], color=DOT_COLOR, s=60, zorder=3,
                       edgecolors='white', linewidths=0.5)
        ax.axhline(1, color='black', lw=1.2, ls='-', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(dow_labels, rotation=0, ha='center')
        ax.set_ylabel('Relative Risk (RR)')
        ax.set_title('Day-of-Week Effect (DLNM)')
        ax.legend(handles=legend_elements, frameon=True, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


# ==========================================================================
# FIT DLNM IN R
# ==========================================================================

# ==========================================================================
# FIT DLNM IN R VIA SUBPROCESS
# ==========================================================================

def fit_dlnm(cfg):
    """
    Fits the DLNM in R by running Rscript.exe as a subprocess.
    All results are written to CSV files in a temp directory and
    read back into Python. Returns a DLNMTrainer.

    Parameters
    ----------
    cfg : dict — see main() for all keys
    """
    tmp = tempfile.mkdtemp().replace('\\', '/')
    p = lambda name: f"{tmp}/{name}"

    lag_max  = cfg['lag_max']
    lag_df   = cfg['lag_df']
    trend_df = cfg['trend_df']

    r_script = f"""
library(dlnm)
library(splines)

# ------------------------------------------------------------------
# Data — use built-in chicagoNMMAPS exactly as in the R script
# ------------------------------------------------------------------
data("chicagoNMMAPS")
data <- chicagoNMMAPS

data$dp01 <- filter(data$dptp, c(1,1)/2, side=1)
data$o301 <- filter(data$o3,   c(1,1)/2, side=1)
data$pm01 <- filter(data$pm10, c(1,1)/2, side=1)

# Set Monday as reference level for dow (matches standard epidemiology convention)
data$dow <- relevel(factor(data$dow), ref='Monday')

# ------------------------------------------------------------------
# Cross-basis — 4 equally spaced interior knots, lag df=5
# ------------------------------------------------------------------
temp_range <- range(data$temp, na.rm=TRUE)
temp_knots <- temp_range[1] + (temp_range[2] - temp_range[1]) / 5 * 1:4

cb_temp <- crossbasis(
    data$temp,
    lag    = {lag_max},
    argvar = list(fun="ns", knots=temp_knots),
    arglag = list(fun="ns", df={lag_df})
)

# ------------------------------------------------------------------
# Fit quasi-Poisson GLM — exactly matching the R script
# ------------------------------------------------------------------
model <- glm(
    death ~ cb_temp + ns(dp01, df=3) + dow + ns(date, df={trend_df}) + o301 + pm01,
    family = quasipoisson(),
    data   = data
)

# ------------------------------------------------------------------
# Fitted values and dispersion
# ------------------------------------------------------------------
write.csv(data.frame(
    y_obs  = model$y,
    mu_hat = model$fitted.values
), "{p('fitted.csv')}", row.names=FALSE)

phi <- summary(model)$dispersion
write.csv(data.frame(phi=phi), "{p('phi.csv')}", row.names=FALSE)

# ------------------------------------------------------------------
# Exposure-lag-response surface
# ------------------------------------------------------------------
pred_temp <- seq(floor(min(data$temp, na.rm=TRUE)),
                 ceiling(max(data$temp, na.rm=TRUE)), 1)

pred <- crosspred(
    cb_temp, model,
    at    = pred_temp,
    bylag = 1,
    cen   = median(data$temp, na.rm=TRUE)
)

write.csv(data.frame(val_seq = pred$predvar),
          "{p('val_seq.csv')}", row.names=FALSE)
write.csv(as.data.frame(pred$matRRfit),
          "{p('per_lag_rr.csv')}", row.names=FALSE)
write.csv(as.data.frame(pred$matRRlow),
          "{p('per_lag_rr_lo.csv')}", row.names=FALSE)
write.csv(as.data.frame(pred$matRRhigh),
          "{p('per_lag_rr_hi.csv')}", row.names=FALSE)
write.csv(data.frame(
    cum_log_rr = log(pred$allRRfit),
    cum_lo     = log(pred$allRRlow),
    cum_hi     = log(pred$allRRhigh)
), "{p('cum_log_rr.csv')}", row.names=FALSE)

# ------------------------------------------------------------------
# Helper: zero crossbasis with correct class/attributes
# ------------------------------------------------------------------
make_cb_zero <- function(n, cb) {{
    m           <- matrix(0, n, ncol(cb))
    colnames(m) <- colnames(cb)
    class(m)    <- class(cb)
    attr(m, "lag")     <- attr(cb, "lag")
    attr(m, "argvar")  <- attr(cb, "argvar")
    attr(m, "arglag")  <- attr(cb, "arglag")
    attr(m, "varname") <- attr(cb, "varname")
    m
}}

# ------------------------------------------------------------------
# Trend — vary date, hold all other predictors at mean/reference
# ------------------------------------------------------------------
n_trend  <- 500
date_seq <- seq(min(data$date), max(data$date), length.out=n_trend)
nd_trend <- data.frame(
    cb_temp = I(make_cb_zero(n_trend, cb_temp)),
    dp01    = rep(mean(data$dp01, na.rm=TRUE), n_trend),
    dow     = factor(rep(levels(data$dow)[1], n_trend), levels=levels(data$dow)),
    date    = date_seq,
    o301    = rep(mean(data$o301, na.rm=TRUE), n_trend),
    pm01    = rep(mean(data$pm01, na.rm=TRUE), n_trend)
)
trend_mat  <- predict(model, newdata=nd_trend, type="terms", se.fit=TRUE)
date_col   <- grep("date", colnames(trend_mat$fit))
trend_term <- trend_mat$fit[, date_col]
trend_se   <- trend_mat$se.fit[, date_col]
trend_term <- trend_term - mean(trend_term)
write.csv(data.frame(mean=trend_term, se=trend_se),
          "{p('trend.csv')}", row.names=FALSE)

# ------------------------------------------------------------------
# Confounders — vary each, hold others at mean/reference
# ------------------------------------------------------------------
for (r_col in c("dp01", "o301", "pm01")) {{
    conf_vals <- seq(
        quantile(data[[r_col]], 0.01, na.rm=TRUE),
        quantile(data[[r_col]], 0.99, na.rm=TRUE),
        length.out=100
    )
    nd <- data.frame(
        cb_temp = I(make_cb_zero(100, cb_temp)),
        dp01    = rep(mean(data$dp01, na.rm=TRUE), 100),
        dow     = factor(rep(levels(data$dow)[1], 100), levels=levels(data$dow)),
        date    = rep(mean(as.numeric(data$date), na.rm=TRUE), 100),
        o301    = rep(mean(data$o301, na.rm=TRUE), 100),
        pm01    = rep(mean(data$pm01, na.rm=TRUE), 100)
    )
    nd[[r_col]] <- conf_vals
    cp <- predict(model, newdata=nd, type="terms", se.fit=TRUE)
    ci <- grep(r_col, colnames(cp$fit))
    ct <- cp$fit[, ci] - mean(cp$fit[, ci])
    cs <- cp$se.fit[, ci]
    write.csv(data.frame(mean=ct, se=cs),
              paste0("{tmp}/", r_col, "_conf.csv"), row.names=FALSE)
}}

# ------------------------------------------------------------------
# Day of week — extract all 7 effects with proper CIs via delta method.
# After mean-centring, uncertainty is propagated through the full
# variance-covariance matrix so all 7 days (including Monday reference)
# have non-zero CIs, matching the DLNAM embedding approach.
# ------------------------------------------------------------------
dow_order  <- c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
coef_names <- names(coef(model))
V          <- vcov(model)

# Build 7-row contrast matrix L: each row selects one day's log-RR
# relative to the intercept. Monday (reference) gets a zero row.
L <- matrix(0, nrow=7, ncol=length(coef(model)))
rownames(L) <- dow_order

for (i in seq_along(dow_order)) {{
    d       <- dow_order[i]
    pattern <- paste0("^dow", d, "$")
    idx     <- grep(pattern, coef_names)
    if (length(idx) == 1) L[i, idx] <- 1
}}

# Raw log-RR estimates for each day (Monday = 0)
raw_log_rr <- as.vector(L %*% coef(model))

# Mean-centre relative to Thursday (week median, used as reference)
n_days       <- 7
thursday_idx <- which(dow_order == 'Thursday')
L_ref        <- L[thursday_idx, , drop=FALSE]
L_centred    <- L - matrix(1, n_days, 1) %*% L_ref

centred_log_rr <- as.vector(L_centred %*% coef(model))
centred_var    <- diag(L_centred %*% V %*% t(L_centred))
centred_se     <- sqrt(centred_var)

write.csv(data.frame(
    day  = dow_order,
    mean = centred_log_rr,
    se   = centred_se
), "{p('dow.csv')}", row.names=FALSE)

cat("DLNM fitting complete.\\n")
"""

    # ------------------------------------------------------------------
    # Write and run R script
    # ------------------------------------------------------------------
    r_script_path = os.path.join(tmp, 'dlnm_fit.R')
    with open(r_script_path, 'w') as f:
        f.write(r_script)

    rscript = cfg.get('rscript_path',
                      r'C:\Program Files\R\R-4.5.0\bin\Rscript.exe')
    print(f"  Running R script via {rscript} ...")
    result = subprocess.run(
        [rscript, '--vanilla', r_script_path],
        capture_output=True,
        text=True,
        encoding='latin-1',   # handles Swedish locale characters
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed with exit code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    # ------------------------------------------------------------------
    # Read results back
    # ------------------------------------------------------------------
    fitted      = pd.read_csv(p('fitted.csv'))
    mu_hat      = fitted['mu_hat'].values
    y_obs       = fitted['y_obs'].values
    phi         = float(pd.read_csv(p('phi.csv'))['phi'].iloc[0])

    val_seq     = pd.read_csv(p('val_seq.csv'))['val_seq'].values
    cum_df      = pd.read_csv(p('cum_log_rr.csv'))
    cum_log_rr  = cum_df['cum_log_rr'].values
    cum_lo      = cum_df['cum_lo'].values
    cum_hi      = cum_df['cum_hi'].values
    per_lag_rr  = pd.read_csv(p('per_lag_rr.csv')).values.T     # (L+1, n_grid)
    per_lag_lo  = pd.read_csv(p('per_lag_rr_lo.csv')).values.T
    per_lag_hi  = pd.read_csv(p('per_lag_rr_hi.csv')).values.T
    lags        = np.arange(per_lag_rr.shape[0], dtype=float)

    ref_idx     = int(np.argmin(np.abs(val_seq - np.median(val_seq))))

    trend       = pd.read_csv(p('trend.csv'))
    trend_rr    = {'mean': trend['mean'].values, 'se': trend['se'].values}

    conf_rr = {}
    for col, r_col in [('dp01', 'dp01'), ('o301', 'o301'), ('pm01', 'pm01')]:
        c = pd.read_csv(p(f'{r_col}_conf.csv'))
        conf_rr[col] = {'mean': c['mean'].values, 'se': c['se'].values}

    dow_df  = pd.read_csv(p('dow.csv'))
    dow_rr  = {'day': dow_df['day'].values,
               'mean': dow_df['mean'].values,
               'se':   dow_df['se'].values}

    # ------------------------------------------------------------------
    # Build trainer wrapper
    # ------------------------------------------------------------------
    trainer = DLNMTrainer(
        mu_hat        = mu_hat,
        cum_rr        = per_lag_rr,
        cum_log_rr    = cum_log_rr,
        cum_lo        = cum_lo,
        cum_hi        = cum_hi,
        per_lag_lo    = per_lag_lo,
        per_lag_hi    = per_lag_hi,
        val_seq       = val_seq,
        lags          = lags,
        ref_idx       = ref_idx,
        trend_rr      = trend_rr,
        conf_rr       = conf_rr,
        dow_rr        = dow_rr,
        phi           = phi,
        exposure_lags = [lag_max],
        device        = torch.device('cpu'),
    )

    return trainer, y_obs


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    cfg = dict(
        rscript_path = r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",

        # Cross-basis specification — matches the R script exactly
        lag_max   = 30,     # maximum lag in days
        lag_df    = 5,      # degrees of freedom for lag basis (ns)

        # GLM specification
        trend_df  = 14 * 7, # df for long-term trend spline ns(date, df=98)

        # Uncertainty intervals
        alpha     = 0.05,
        ci_type   = 'wald',
        centering = 'median',
    )

    # ------------------------------------------------------------------
    # Fit DLNM in R
    # ------------------------------------------------------------------
    trainer, y_obs = fit_dlnm(cfg)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluator = PerformanceEvaluator(trainer)
    Y         = torch.FloatTensor(y_obs).unsqueeze(1)
    X_dummy   = [torch.zeros(len(y_obs), 1)]
    X_c_dummy = torch.zeros(len(y_obs), 1)
    X_t_dummy = torch.zeros(len(y_obs), 1)

    metrics = evaluator.calculate_metrics(
        X_dummy, X_c_dummy, X_t_dummy, Y,
        alpha   = cfg['alpha'],
        ci_type = cfg['ci_type'],
    )
    evaluator.print_report(metrics)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    class _DummyProcessor:
        class _IdentityScaler:
            def transform(self, df):
                return df.values
        scalers = {'temp': _IdentityScaler()}

    viz = DLNMVisualizer(
        trainer,
        _DummyProcessor(),
        alpha     = cfg['alpha'],
        ci_type   = cfg['ci_type'],
        centering = cfg['centering'],
        phi       = trainer.phi,
    )

    # Minimal data frame for axis labels — just the val_seq range
    data = pd.DataFrame({'temp': trainer.val_seq})

    viz.plot_all(
        data,
        exposure_cols = ['temp'],
        conf_cols     = ['dp01', 'o301', 'pm01'],
        X_encodings      = None,
        encoding_configs = [],
    )

    return trainer, metrics


if __name__ == '__main__':
    main()
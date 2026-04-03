# evaluation.py

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PerformanceEvaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def calculate_metrics(self, X_exposures, X_c, X_time, Y,
                          alpha=0.05,
                          ci_type='ensemble',
                          x_encodings=None):
        """
        alpha        : significance level (e.g. 0.05 for 95% intervals)
        ci_type      : 'ensemble' | 'poisson' | 'wald'
        x_encodings  : list of LongTensors, one per categorical encoding, or None
        """
        device   = self.trainer.device
        Y_tensor = Y if isinstance(Y, torch.Tensor) else torch.tensor(Y)

        X_exp_d  = [x.to(device) for x in X_exposures]
        X_c_d    = X_c.to(device)
        X_time_d = X_time.to(device)
        enc_d    = [e.to(device) for e in x_encodings] if x_encodings else None

        all_preds = []
        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.eval()
                pred = model(X_exp_d, X_c_d, X_time_d, enc_d).cpu().numpy().flatten()
                all_preds.append(pred)

        all_preds = np.array(all_preds)
        Y_true    = Y_tensor.cpu().numpy().flatten()
        mu_hat    = np.mean(all_preds, axis=0)

        phi = float(np.mean((Y_true - mu_hat) ** 2 / (mu_hat + 1e-8)))
        z   = stats.norm.ppf(1.0 - alpha / 2.0)

        # Ensemble interval
        sigma_hat = np.std(all_preds, axis=0)
        lo_ens    = mu_hat - z * sigma_hat
        hi_ens    = mu_hat + z * sigma_hat

        # Poisson interval (exact quantiles)
        lo_pois   = stats.poisson.ppf(alpha / 2.0,       mu=mu_hat)
        hi_pois   = stats.poisson.ppf(1.0 - alpha / 2.0, mu=mu_hat)

        # Wald interval (quasi-Poisson)
        se_wald   = np.sqrt(phi * mu_hat)
        lo_wald   = mu_hat - z * se_wald
        hi_wald   = mu_hat + z * se_wald

        def _coverage(lo, hi):
            return float(np.mean((Y_true >= lo) & (Y_true <= hi)))

        rmse      = np.sqrt(mean_squared_error(Y_true, mu_hat))
        null_pred = np.full_like(Y_true, np.mean(Y_true))
        null_rmse = np.sqrt(mean_squared_error(Y_true, null_pred))
        null_mae  = mean_absolute_error(Y_true, null_pred)

        return {
            "RMSE":              rmse,
            "MAE":               mean_absolute_error(Y_true, mu_hat),
            "R2":                r2_score(Y_true, mu_hat),
            "Null_RMSE":         null_rmse,
            "Null_MAE":          null_mae,
            "Phi":               phi,
            "alpha":             alpha,
            "ci_type":           ci_type,
            "z":                 z,
            "Coverage_Ensemble": _coverage(lo_ens,  hi_ens),
            "Coverage_Poisson":  _coverage(lo_pois, hi_pois),
            "Coverage_Wald":     _coverage(lo_wald, hi_wald),
            "Nominal_Coverage":  1.0 - alpha,
            "_lo_ens":  lo_ens,  "_hi_ens":  hi_ens,
            "_lo_pois": lo_pois, "_hi_pois": hi_pois,
            "_lo_wald": lo_wald, "_hi_wald": hi_wald,
            "_mu_hat":  mu_hat,
            "_Y_true":  Y_true,
        }

    def print_report(self, m):
        sep     = "=" * 60
        nominal = int(round(m['Nominal_Coverage'] * 100))

        print(f"\n{sep}")
        print(f"  DLNAM PERFORMANCE REPORT")
        print(f"{sep}")
        print(f"  R²:          {m['R2']:.4f}")
        print(f"  RMSE:        {m['RMSE']:.2f}")
        print(f"  Null RMSE:   {m['Null_RMSE']:.2f}")
        print(f"  MAE:         {m['MAE']:.2f}")
        print(f"  Null MAE:    {m['Null_MAE']:.2f}")
        print(f"  Phi:         {m['Phi']:.4f}")
        print(f"{sep}")
        print(f"  Coverage rates:  {nominal}% Confidence Intervals")
        print(f"    Ensemble:  {m['Coverage_Ensemble']*100:.2f}%")
        print(f"    Poisson:   {m['Coverage_Poisson']*100:.2f}%")
        print(f"    Wald:      {m['Coverage_Wald']*100:.2f}%")
        print(f"{sep}\n")
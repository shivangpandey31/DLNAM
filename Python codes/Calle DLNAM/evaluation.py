# evaluation.py

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PerformanceEvaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def calculate_metrics(self, X_exposures, X_c, X_time, Y):
        Y_true = Y.cpu().numpy().flatten()
        all_preds = []

        with torch.no_grad():
            for model in self.trainer.ensemble:
                model.eval()
                pred = model(X_exposures, X_c, X_time).cpu().numpy().flatten()
                all_preds.append(pred)

        Y_pred = np.mean(all_preds, axis=0)

        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))

        # Null model: constant prediction at the observed mean
        null_pred = np.full_like(Y_true, np.mean(Y_true))
        null_rmse = np.sqrt(mean_squared_error(Y_true, null_pred))

        return {
            "RMSE": rmse,
            "MAE": mean_absolute_error(Y_true, Y_pred),
            "R2": r2_score(Y_true, Y_pred),
            "Null_RMSE": null_rmse,
            "Pct_Improvement": ((null_rmse - rmse) / null_rmse) * 100,
        }

    def print_report(self, m):
        sep = "=" * 40
        print(f"\n{sep}\n MASTER'S THESIS: DLNAM REPORT \n{sep}")
        print(f"R² Score:    {m['R2']:.4f}")
        print(f"RMSE:        {m['RMSE']:.2f}")
        print(f"Null RMSE:   {m['Null_RMSE']:.2f}")
        print(f"Improvement: {m['Pct_Improvement']:.2f}%")
        print(sep)
"""
dlnam_bench/export.py — write simulated datasets + a manifest for the R DLNM fit.

Seeds match MonteCarloStudy (base_seed + rep) so the DLNM and the DLNAM are
fit to the SAME simulated datasets (paired comparison). Each dataset CSV holds
the full exposure series and the outcome with NA on the lag-padding rows, so R's
crossbasis + glm drop exactly those rows.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from dlnam_sim.dgp import DataGeneratingProcess


def export_datasets(
    scenarios: Dict[str, DataGeneratingProcess],
    out_dir: str,
    n_reps: int,
    n_obs: int,
    grid: np.ndarray,
    reference: float,
    alpha: float = 0.05,
    base_seed: int = 0,
    value_df_grid: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    lag_df_grid: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    penalized_value_df=None,
    penalized_lag_df=None,
    tdlnm_burn: int = 5000,
    tdlnm_iter: int = 15000,
    tdlnm_thin: int = 5,
    tdlnm_attempts: int = 3,
    tdlnm_exposure_splits: int = 20,
) -> str:
    """Write data/<scenario>_rep###.csv for every (scenario, replicate) and a
    manifest.json describing the grid, reference, lag, QAIC search space, and the
    expected R output paths. Returns the manifest path."""
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)

    # all scenarios share exposure/target/lag conventions; read from the first
    any_dgp = next(iter(scenarios.values()))
    exposure_col = next(iter(any_dgp.covariate_sampler))
    target_col = any_dgp.target_col
    lag_max = any_dgp.true_terms[exposure_col].lag_max
    pen_vdf = int(penalized_value_df if penalized_value_df is not None
                  else max(value_df_grid))
    pen_ldf = int(penalized_lag_df if penalized_lag_df is not None
                  else max(lag_df_grid))

    datasets = []
    for scenario, dgp in scenarios.items():
        for rep in range(n_reps):
            seed = base_seed + rep
            sim = dgp.simulate(n_obs, seed)
            frame = sim.frame
            # Export the full exposure series + outcome. R's crossbasis yields NA
            # for the first lag_max rows and glm(na.omit) drops them -- exactly the
            # DGP's lag-padding rows (total_lag == lag_max for these scenarios), so
            # no fake-outcome rows enter the fit.
            out_frame = pd.DataFrame({exposure_col: frame[exposure_col].values,
                                      target_col: frame[target_col].values})
            rel = f"data/{scenario}_rep{rep:03d}.csv"
            out_frame.to_csv(os.path.join(out_dir, rel), index=False)
            datasets.append({
                "scenario": scenario, "rep": rep, "data": rel,
                "cumulative": f"out/{scenario}_rep{rep:03d}_cum.csv",
                "surface":    f"out/{scenario}_rep{rep:03d}_surf.csv",
            })

    manifest = {
        "exposure_col": exposure_col, "target_col": target_col,
        "lag_max": int(lag_max), "reference": float(reference),
        "alpha": float(alpha), "ci_level": float(1 - alpha),
        "grid": [float(v) for v in np.asarray(grid)],
        "value_df_grid": list(value_df_grid), "lag_df_grid": list(lag_df_grid),
        "penalized_value_df": pen_vdf, "penalized_lag_df": pen_ldf,
        "tdlnm_burn": int(tdlnm_burn), "tdlnm_iter": int(tdlnm_iter),
        "tdlnm_thin": int(tdlnm_thin), "tdlnm_attempts": int(tdlnm_attempts),
        "tdlnm_exposure_splits": int(tdlnm_exposure_splits),
        "n_obs": int(n_obs), "n_reps": int(n_reps), "base_seed": int(base_seed),
        "datasets": datasets,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return mpath

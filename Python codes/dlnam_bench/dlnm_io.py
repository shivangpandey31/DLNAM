"""
dlnam_bench/dlnm_io.py — load R's exported DLNM curves into a StudyResult.

The R script writes one cumulative-exposure-response CSV per (scenario, rep)
with columns: value, fit, lo, hi (response-scale RR, centered at the manifest
reference). This loads them, attaches the DGP truth on the same grid, and
returns the SAME StudyResult type the DLNAM Monte Carlo produces — so bias,
RMSE, coverage and error_mean_se are computed identically for both models.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from dlnam import make_link
from dlnam.terms.base import Centering
from dlnam_sim.dgp import DataGeneratingProcess
from dlnam_sim.study import StudyResult, ReplicateResult


def _truth_on_grid(dgp, term, grid, centering, link):
    curve = dgp.truth_curve(term, grid, centering)
    if link.name in ("log", "logit"):
        return np.exp(curve.log_effect)
    return curve.log_effect


def load_dlnm_study(out_dir: str, dgp: DataGeneratingProcess, scenario: str,
                    centering: Centering, term: str = "x",
                    prefix: str = "") -> StudyResult:
    """Assemble a StudyResult for the DLNM fits of one scenario. `prefix` selects
    the method's output files ("qaic_", "qbic_", "pen_", "tree_"). Skips
    replicates whose R output file is missing (so it works on a partial R run)."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    grid = np.asarray(manifest["grid"], dtype=float)
    link = make_link("log")
    truth = _truth_on_grid(dgp, term, grid, centering, link)

    def _pre(path):
        d, b = os.path.split(path)
        return os.path.join(d, prefix + b)

    result = StudyResult(truth={term: truth}, grids={term: grid})
    for d in manifest["datasets"]:
        if d["scenario"] != scenario:
            continue
        cpath = os.path.join(out_dir, _pre(d["cumulative"]))
        if not os.path.exists(cpath):
            continue
        df = pd.read_csv(cpath)
        est = {term: {"mean": df["fit"].values,
                      "lo": df["lo"].values, "hi": df["hi"].values}}
        result.replicates.append(ReplicateResult(seed=d["rep"], estimates=est))
    return result

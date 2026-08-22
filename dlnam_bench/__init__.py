"""
dlnam_bench -- model-comparison harness (DLNAM vs DLNM-family methods).

Separate from both `dlnam` (the model) and `dlnam_sim` (DGPs + Monte Carlo):
it depends on both but neither imports it. Workflow:

  1. export_datasets(...)        # Python writes simulated data + a manifest
  2. Rscript dlnam_bench/fit_dlnm.R <out_dir>   # R fits DLNM/TDLNM curves
  3. load_dlnm_study(...)        # Python loads R's curves as a StudyResult

The R-side curves are then scored by the SAME StudyResult logic as the DLNAM, so
the comparison differs only in the estimator, not in centering/grid/scoring.
"""
from .export import export_datasets
from .dlnm_io import load_dlnm_study, load_dlnm_surface_study

__all__ = ["export_datasets", "load_dlnm_study", "load_dlnm_surface_study"]

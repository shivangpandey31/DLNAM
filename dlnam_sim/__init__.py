"""
dlnam_sim — simulation & Monte-Carlo validation add-on.

Depends on `dlnam`; `dlnam` does NOT depend on this. Safe to ship, install, or
delete independently of the core. Folding this into `dlnam.simulation` as a
subpackage instead would be equally valid as long as the one-way import rule
holds.
"""

from .dgp import (
    DataGeneratingProcess, SimulatedDataset, FunctionTerm,
    Sampler, PoissonSampler, BernoulliSampler, NegBinSampler,
)
from .study import MonteCarloStudy, StudyResult, ReplicateResult

__all__ = [
    "DataGeneratingProcess", "SimulatedDataset", "FunctionTerm",
    "Sampler", "PoissonSampler", "BernoulliSampler", "NegBinSampler",
    "MonteCarloStudy", "StudyResult", "ReplicateResult",
]
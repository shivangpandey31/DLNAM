"""Per-replicate DLNAM error for the joint MC, with no comparator fits.

The joint bundle stores only summary statistics, so a large variance term cannot
be told apart from a handful of diverged replicates. This runs the DLNAM arm
alone and reports each replicate's RMSE, which distinguishes the two directly.

The R comparators are skipped entirely: they do not depend on the training
budget, so they carry no information for this question and account for roughly
two thirds of the runtime.

usage:
    python experiments/diag_joint_replicates.py EPOCHS N_REPS [OUT.json]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from dlnam.config import TrainConfig
from dlnam.terms.base import Centering
from dlnam_sim.study import MonteCarloStudy

import run_mc_joint as J
from dlnam_sim import scenarios_joint as MX

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
N_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = sys.argv[3] if len(sys.argv) > 3 else f"joint_replicates_e{EPOCHS}_r{N_REPS}.json"

torch.manual_seed(J.SEED)
np.random.seed(J.SEED)

dgp = MX.joint_dgp(lag_max=J.LAG, include_null=J.INCLUDE_NULL)
cen = Centering(method="reference", value=J.REF)
tcfg = TrainConfig(epochs=EPOCHS, n_ensemble=J.N_ENSEMBLE, lr=8e-4, lr_min=1e-4,
                   weight_decay=1e-4, schedule="cosine", grad_clip=10)

print(f"DLNAM only: epochs={EPOCHS}  n_reps={N_REPS}  seed={J.SEED}  "
      f"device={J.DEVICE}", flush=True)

study = MonteCarloStudy(
    dgp=dgp,
    model_config=J.dlnam_config(J.LAG),
    train_config=tcfg,
    centering=cen,
    n_reps=N_REPS,
    n_obs=J.N_OBS,
    base_seed=J.SEED,
    device=J.DEVICE,
    se_source="laplace+ensemble",
).run(progress=True)

terms = [t for t in study.truth]
rows = {}
print(f"\n{'rep':>4} {'seed':>5} " + " ".join(f"{t:>9}" for t in terms))
for i, rep in enumerate(study.replicates):
    per = {}
    for t in terms:
        est = np.log(np.asarray(rep.estimates[t]["mean"], dtype=float))
        tru = np.log(np.asarray(study.truth[t], dtype=float))
        per[t] = float(np.sqrt(((est - tru) ** 2).mean()))
    rows[i] = {"seed": int(rep.seed), "rmse": per}
    print(f"{i:>4} {rep.seed:>5} " + " ".join(f"{per[t]:>9.4f}" for t in terms),
          flush=True)

print(f"\n{'term':>10} {'median':>9} {'mean':>9} {'max':>9} {'max/median':>11}")
for t in terms:
    v = np.array([rows[i]["rmse"][t] for i in rows])
    med = float(np.median(v))
    print(f"{t:>10} {med:>9.4f} {v.mean():>9.4f} {v.max():>9.4f} "
          f"{v.max() / max(med, 1e-12):>11.1f}")

print("\nA few diverged fits show as max/median >> 1 with the median near the "
      "single-replicate value; broad instability shows as a high median.")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"epochs": EPOCHS, "n_reps": N_REPS, "seed": J.SEED,
               "terms": terms, "replicates": rows}, fh, indent=1)
print(f"wrote {OUT}")

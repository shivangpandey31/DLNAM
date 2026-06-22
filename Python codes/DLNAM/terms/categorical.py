"""
terms/categorical.py — categorical term.

Embedding-based (not the wasteful one-hot + matmul of the original): with an
empty hidden stack it is a pure lookup table (one learned scalar per level);
with hidden layers it is Embedding -> MLP -> scalar. The embedding is zero-
initialised so the term starts at no effect, and the global level is carried by
the model intercept (the reference level is absorbed there).

Centering: the "grid" is the category indices and the natural anchor is a
reference *level* rather than a continuous value. Setting _data_median to the
middle index makes the default 'median' centering reproduce the original DLNM
convention (e.g. Thursday for a Mon-Sun encoding). A specific reference level is
obtained with Centering(method='reference', value=<level index>).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import CategoricalTermSpec
from .base import AdditiveTerm


class CategoricalTerm(AdditiveTerm):
    def __init__(self, name: str, spec: CategoricalTermSpec):
        super().__init__(name, scaling="none")
        self.spec = spec
        C = spec.num_categories
        self.num_categories = C
        self.order = list(spec.order) if spec.order else [str(i) for i in range(C)]
        layers = list(spec.layers)

        if not layers:                                   # pure lookup table
            self.emb = nn.Embedding(C, 1)
            self.tail: nn.Module = nn.Identity()
        else:
            self.emb = nn.Embedding(C, layers[0].width)
            mods = [layers[0].activation.build()]
            last = layers[0].width
            for ls in layers[1:]:
                mods += [nn.Linear(last, ls.width), ls.activation.build()]
                last = ls.width
            mods.append(nn.Linear(last, 1))
            self.tail = nn.Sequential(*mods)

        nn.init.zeros_(self.emb.weight)                  # start at no effect
        self._value_range = (0.0, float(C - 1))
        self._data_median = float(C // 2)                # middle level = reference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) long category indices  -> (B, 1)
        return self.tail(self.emb(x.long()))

    def default_grid(self, n: int = None) -> np.ndarray:
        return np.arange(self.num_categories, dtype=float)

    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        idx = np.asarray(grid_raw).round().astype(int)
        x = torch.tensor(idx, dtype=torch.long, device=self.emb.weight.device)
        with torch.no_grad():
            return self.tail(self.emb(x)).squeeze(-1).cpu().numpy()

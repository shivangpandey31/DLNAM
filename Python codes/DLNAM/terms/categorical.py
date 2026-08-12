"""
terms/categorical.py — categorical additive terms.

Two encodings are supported for compatibility with the original DLNAM:

* ``encoding_type='one_hot'``: explicitly builds a one-hot design and applies
  either a single linear lookup (``layers=()``) or an MLP.
* ``encoding_type='embedding'``: memory-efficient lookup. With
  ``embedding_dim=1`` and ``layers=()`` this is a scalable fixed-effect style
  person/category intercept and is mathematically equivalent to a linear
  lookup on a one-hot design, without materialising the one-hot matrix.

The category ``order`` may be supplied in the config. If it is empty,
DataProcessor infers a deterministic order from the data and calls
``set_category_order`` before tensors are created. ``num_categories`` must still
be known when the model is built because it determines parameter dimensions.
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
        self.num_categories = int(spec.num_categories)
        self.encoding_type = spec.encoding_type
        self.embedding_dim = int(spec.embedding_dim)
        self.source_col = spec.source_col or name
        self.role = spec.role
        self.order = list(spec.order) if spec.order else [str(i) for i in range(self.num_categories)]
        layers = list(spec.layers)

        if self.encoding_type not in ("one_hot", "embedding"):
            raise ValueError(
                "CategoricalTermSpec.encoding_type must be 'one_hot' or 'embedding'"
            )

        if self.encoding_type == "one_hot":
            self.emb = None
            if not layers:
                # Pure categorical lookup / fixed-effect style term.
                self.tail = nn.Linear(self.num_categories, 1, bias=(self.role != "strata"))
                nn.init.zeros_(self.tail.weight)
                if self.tail.bias is not None:
                    nn.init.zeros_(self.tail.bias)
            else:
                mods = []
                first = layers[0]
                lin = nn.Linear(self.num_categories, first.width)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                mods.extend([lin, first.activation.build()])
                if first.dropout > 0:
                    mods.append(nn.Dropout(first.dropout))
                last = first.width
                for ls in layers[1:]:
                    mods.extend([nn.Linear(last, ls.width), ls.activation.build()])
                    if ls.dropout > 0:
                        mods.append(nn.Dropout(ls.dropout))
                    last = ls.width
                mods.append(nn.Linear(last, 1))
                self.tail = nn.Sequential(*mods)

        else:  # embedding
            self.emb = nn.Embedding(self.num_categories, self.embedding_dim)
            nn.init.zeros_(self.emb.weight)
            if self.embedding_dim == 1 and not layers:
                self.tail = nn.Identity()
            else:
                mods = []
                last = self.embedding_dim
                for ls in layers:
                    mods.extend([nn.Linear(last, ls.width), ls.activation.build()])
                    if ls.dropout > 0:
                        mods.append(nn.Dropout(ls.dropout))
                    last = ls.width
                mods.append(nn.Linear(last, 1))
                self.tail = nn.Sequential(*mods)

        self._value_range = (0.0, float(self.num_categories - 1))
        self._data_median = float(self.num_categories // 2)

    def set_category_order(self, order) -> None:
        """Attach real category labels after DataProcessor infers them.

        This is deliberately metadata-only: the network dimensions were fixed
        at construction by ``num_categories``. The processor validates that the
        inferred order has exactly that many levels before calling this method.
        """
        order = list(order)
        if len(order) != self.num_categories:
            raise ValueError(
                f"'{self.name}' expected {self.num_categories} categories but "
                f"received an order with {len(order)} levels"
            )
        self.order = order
        self._value_range = (0.0, float(self.num_categories - 1))
        self._data_median = float(self.num_categories // 2)

    def _one_hot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            x.long().reshape(-1), num_classes=self.num_categories
        ).to(dtype=next(self.parameters()).dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long().reshape(-1)
        if self.encoding_type == "one_hot":
            out = self.tail(self._one_hot(x))
            if self.role == "strata" and isinstance(self.tail, nn.Linear):
                # Sum-to-zero centring removes the arbitrary intercept shift.
                out = out - self.tail.weight.mean()
            return out

        z = self.emb(x)
        if self.role == "strata" and isinstance(self.tail, nn.Identity):
            # One learned scalar per stratum, centred so the global intercept
            # retains its usual interpretation.
            z = z - self.emb.weight.mean(dim=0, keepdim=True)
        return self.tail(z)

    def _last_layer_design(self, x: torch.Tensor) -> torch.Tensor:
        """Exact design for the categorical term's final linear parameters."""
        x = x.long().reshape(-1)

        if self.encoding_type == "one_hot":
            oh = self._one_hot(x)
            if isinstance(self.tail, nn.Linear):
                # The whole term is the final linear layer. Strata coefficients
                # are represented with a sum-to-zero centred design and no bias.
                if self.role == "strata":
                    return oh - (1.0 / self.num_categories)
                if self.tail.bias is None:
                    return oh
                return torch.cat([
                    oh,
                    torch.ones((len(x), 1), dtype=oh.dtype, device=oh.device),
                ], dim=1)
            feat = self.tail[:-1](oh)
        else:
            if isinstance(self.tail, nn.Identity):
                # Scalar embedding lookup: one parameter per category.
                design = torch.nn.functional.one_hot(
                    x, num_classes=self.num_categories
                ).to(dtype=self.emb.weight.dtype)
                if self.role == "strata":
                    design = design - (1.0 / self.num_categories)
                return design
            feat = self.tail[:-1](self.emb(x))

        return torch.cat([
            feat,
            torch.ones((len(x), 1), dtype=feat.dtype, device=feat.device),
        ], dim=1)

    def default_grid(self, n: int = None) -> np.ndarray:
        return np.arange(self.num_categories, dtype=float)

    def raw_log_effect(self, grid_raw: np.ndarray) -> np.ndarray:
        idx = np.asarray(grid_raw).round().astype(int)
        device = next(self.parameters()).device
        x = torch.tensor(idx, dtype=torch.long, device=device)
        with torch.no_grad():
            return self.forward(x).squeeze(-1).cpu().numpy()

"""
activations.py — activation construction and output transforms.

`build_activation(spec)` turns an ActivationSpec into a concrete module,
optionally composed with an output transform (e.g. the learnable soft-cap).
Kept separate from config.py so config stays import-light; config's
`.build()` methods defer here via a local import.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(y: float) -> float:
    # softplus(x) = log(1+exp(x)); invert so SoftCap starts exactly at init_cap.
    import math
    return math.log(math.expm1(y))


class SoftCap(nn.Module):
    """Smoothly bounds outputs to (-cap, +cap) via cap * tanh(x / cap).

    cap = softplus(raw) stays positive; `learnable` makes it a trained
    parameter. As cap -> inf this approaches the identity, so it degrades
    gracefully toward "no cap".
    """

    def __init__(self, init_cap: float = 8.0, learnable: bool = True):
        super().__init__()
        raw = torch.tensor(float(_inverse_softplus(init_cap)))
        if learnable:
            self.raw_cap = nn.Parameter(raw)
        else:
            self.register_buffer("raw_cap", raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cap = F.softplus(self.raw_cap)
        return cap * torch.tanh(x / cap)


def build_transform(spec) -> nn.Module:
    from .config import SoftCapSpec
    if isinstance(spec, SoftCapSpec):
        return SoftCap(init_cap=spec.init_cap, learnable=spec.learnable)
    raise TypeError(f"Unknown transform spec: {type(spec).__name__}")


def build_activation(spec) -> nn.Module:
    base = spec.base()
    if spec.transform is None:
        return base
    return nn.Sequential(base, build_transform(spec.transform))

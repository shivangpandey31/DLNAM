"""
layers.py — low-level building blocks shared across terms.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def init_linear(linear, weight_init=None, bias_init=None):
    """Apply optional InitSpec to an nn.Linear's weight/bias in place; leaves
    PyTorch defaults when None. The 'torch_linear' bias scheme reproduces
    PyTorch's fan-in-based default exactly (needs the layer's in_features)."""
    if weight_init is not None:
        weight_init.apply_(linear.weight)
    if bias_init is not None and linear.bias is not None:
        if getattr(bias_init, "scheme", None) == "torch_linear":
            import math
            fan_in = linear.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(linear.bias, -bound, bound)
        else:
            bias_init.apply_(linear.bias)
    return linear


def init_mix(spec):
    """Subnet mixing-weight init: spec.mix_init if given, else constant 1/S
    (contributions sum to ~1 at init; aligns with single-subnet DLNAM at S=1)."""
    S = spec.num_subnets
    t = torch.empty(S)
    if getattr(spec, "mix_init", None) is not None:
        spec.mix_init.apply_(t)
    else:
        nn.init.constant_(t, 1.0 / S)
    return t


def apply_subnet_dropout(weights: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Apply inverted dropout to subnet mixing weights during training."""
    p = float(p)
    if not training or p <= 0.0 or weights.numel() <= 1:
        return weights
    keep = 1.0 - p
    mask = (torch.rand_like(weights) < keep).to(weights.dtype)
    if torch.count_nonzero(mask) == 0:
        idx = torch.randint(mask.numel(), (), device=mask.device)
        mask[idx] = 1.0
    return weights * mask / keep


def _init_per_dim_weight(in_features, out_features, means, std):
    """Weight (in_features, out_features); row i (input dim i) initialised from
    N(means[i], std). `means` may be a scalar or a length-in_features sequence,
    which is how per-dimension init means mu_{w_(i)} are realised inside a single
    shared weight matrix."""
    if np.isscalar(means):
        means = [float(means)] * in_features
    W = torch.empty(in_features, out_features)
    for i in range(in_features):
        nn.init.normal_(W[i], mean=float(means[i]), std=std)
    return W


class ExULayer(nn.Module):
    """Exp-centred unit (Agarwal et al., Neural Additive Models).

    Computes, per output unit j:  sum_i exp(w_ij) * (x_i - b_j)
    Bias is per-UNIT (b_j), shared across input dims. The concatenation encoder
    uses this with in_features=1 (one scalar ExU per input dimension).

    weight_mean / weight_std parameterise the log-domain weight init; larger
    means -> sharper initial features.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_mean: float = 1.5, weight_std: float = 0.5, bias_init=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.normal(weight_mean, weight_std, size=(in_features, out_features))
        )
        b = torch.empty(out_features).uniform_(0.0, 1.0)
        if bias_init is not None: bias_init.apply_(b)
        self.bias = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features) -> (N, out_features)
        shifted = x.unsqueeze(-1) - self.bias          # (N, in, out)
        return torch.sum(shifted * torch.exp(self.weight), dim=1)


class ExULayerSharedBias(nn.Module):
    """Unified multi-dim ExU with one bias per INPUT DIMENSION, shared across
    units:  h_o = sum_i exp(W_io) * (x_i - b_i),  b in R^{k0}.

    This encoder does not reduce to the scalar ExU at k0=1 because the bias is
    shared across units rather than local to each unit.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_means, weight_std: float = 0.5, bias_init=None):
        super().__init__()
        self.weight = nn.Parameter(
            _init_per_dim_weight(in_features, out_features, weight_means, weight_std))
        b = torch.empty(in_features).uniform_(0.0, 1.0)
        if bias_init is not None: bias_init.apply_(b)
        self.bias = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in) ;  (x - b) @ exp(W)  -> (N, out)
        return (x - self.bias) @ torch.exp(self.weight)


class ExULayerLocalBias(nn.Module):
    """Unified multi-dim ExU with a bias per UNIT and per INPUT DIMENSION:
    h_o = sum_i exp(W_io) * (x_i - B_io),  B in R^{k0 x k1}.

    This is the paper's second alternative h = sigma((e^W ⊙ (1 x^T - B)) 1).
    It preserves per-unit localisation and reduces exactly to the scalar ExU at
    k0=1, while mixing dimensions already in the input layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_means, weight_std: float = 0.5, bias_init=None):
        super().__init__()
        self.weight = nn.Parameter(
            _init_per_dim_weight(in_features, out_features, weight_means, weight_std))
        b = torch.empty(in_features, out_features).uniform_(0.0, 1.0)
        if bias_init is not None: bias_init.apply_(b)
        self.bias = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x.unsqueeze(-1) - self.bias          # (N, in, out)
        return torch.sum(shifted * torch.exp(self.weight), dim=1)

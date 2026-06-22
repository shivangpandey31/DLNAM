"""
links.py — link functions, each owning its matching loss.

The link is decoupled from the model's forward pass (the model returns the
linear predictor eta; the link maps eta -> mu). Crucially, a Link knows which
loss it pairs with, so an invalid combination (logit link + Poisson loss) is
impossible to construct rather than discovered as a NaN mid-training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Link(ABC):
    name: str

    @abstractmethod
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """Map linear predictor eta to mean mu (response scale)."""

    @abstractmethod
    def make_loss(self) -> nn.Module:
        """Return the negative log-likelihood module matched to this link."""

    @abstractmethod
    def effect_to_response(self, log_or_logit_effect: torch.Tensor) -> torch.Tensor:
        """Map a centered additive effect to the reported response-scale
        quantity (RR for log link, OR for logit link). Used by inference so
        the effect-scale is defined in exactly one place."""


class LogLink(Link):
    name = "log"

    def inverse(self, eta):
        return torch.exp(eta)              # soft-capping handled upstream by a transform

    def make_loss(self):
        return PoissonNLL()

    def effect_to_response(self, effect):
        return torch.exp(effect)           # relative risk


class LogitLink(Link):
    name = "logit"

    def inverse(self, eta):
        return torch.sigmoid(eta)

    def make_loss(self):
        return BernoulliNLL()

    def effect_to_response(self, effect):
        return torch.exp(effect)           # odds ratio


class IdentityLink(Link):
    name = "identity"

    def inverse(self, eta):
        return eta

    def make_loss(self):
        return GaussianNLL()

    def effect_to_response(self, effect):
        return effect


# ---------------------------------------------------------------------------
# Losses (kept here so the link/loss pairing lives in one file)
# ---------------------------------------------------------------------------

class PoissonNLL(nn.Module):
    def forward(self, mu, y):
        return torch.mean(mu - y * torch.log(mu + 1e-8))


class BernoulliNLL(nn.Module):
    def forward(self, mu, y):
        mu = torch.clamp(mu, 1e-7, 1.0 - 1e-7)
        return torch.mean(-y * torch.log(mu) - (1.0 - y) * torch.log(1.0 - mu))


class GaussianNLL(nn.Module):
    def forward(self, mu, y):
        return torch.mean((mu - y) ** 2)


_LINKS = {"log": LogLink, "logit": LogitLink, "identity": IdentityLink}


def make_link(name: str) -> Link:
    return _LINKS[name]()

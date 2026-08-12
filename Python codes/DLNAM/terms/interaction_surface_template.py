"""
INTERACTION SURFACE ROADMAP (non-imported template)
==================================================

This file is intentionally NOT imported by DLNAM/__init__.py and contains no
active model code. It documents the minimal changes needed to add a compound-
extreme term later without changing current v2 behaviour.

Target estimand
---------------
For two climate variables x1 and x2, use a same-lag interaction surface

    f_interaction(x1[t-l], x2[t-l], l)

and add it to the existing main effects, e.g.

    eta = intercept
          + f_tas(tas, lag)
          + f_hurs(hurs, lag)
          + f_tas_hurs(tas, hurs, lag)

Keeping the two main SurfaceTerms is recommended. A later identifiability step
should constrain/centre the interaction so it represents deviation from the
additive main-effects model rather than relearning the main effects.

Suggested implementation points
-------------------------------

1) config.py

# @dataclass(frozen=True)
# class InteractionSurfaceTermSpec(TermSpec):
#     exposures: tuple[str, str] = ("tas", "hurs")
#     lag_max: int = 21
#     input_exu: Optional[ExUSpec] = field(default_factory=ExUSpec)
#     interaction_mode: Literal["same_lag"] = "same_lag"
#
#     def __post_init__(self):
#         super().__post_init__()
#         if len(self.exposures) != 2:
#             raise ValueError("initial interaction implementation expects 2 exposures")
#         if len(self.layers) == 0:
#             raise ValueError("InteractionSurfaceTermSpec requires LayerSpec entries")

2) terms/interaction_surface.py

# class InteractionSurfaceTerm(AdditiveTerm):
#     # Input is a tuple/dict containing TWO already-prepared lag matrices.
#
#     tas.shape  == (B, lag_max+1)
#     hurs.shape == (B, lag_max+1)
#
#     For each lag l, construct [tas_l, hurs_l, normalised_lag] and evaluate a
#     3-input neural surface. Sum the per-lag contributions exactly as the
#     current SurfaceTerm does.
#
#     def forward(self, inputs):
#         x1, x2 = inputs
#         B, L = x1.shape
#         lag = self.lag_grid.to(x1.device).repeat(B).unsqueeze(1)
#         xyz = torch.cat([
#             x1.reshape(-1, 1),
#             x2.reshape(-1, 1),
#             lag,
#         ], dim=1)
#         point_effect = self._mixed(xyz)
#         return point_effect.view(B, L).sum(dim=1, keepdim=True)

3) model.py

The current forward route assumes every term consumes inputs[name]. An
interaction term must instead declare which existing prepared inputs it uses.
A minimal dispatch would be:

# if isinstance(term.spec, InteractionSurfaceTermSpec):
#     x = tuple(inputs[n] for n in term.spec.exposures)
#     contribution = term(x)
# else:
#     contribution = term(inputs[name])

and add InteractionSurfaceTermSpec -> InteractionSurfaceTerm to from_config().

4) data.py

NO new interaction matrix is necessary. The processor should continue preparing
ordinary lag matrices for tas and hurs. The interaction term reuses those two
matrices, which avoids duplicating a potentially enormous B x lag tensor.

5) inference.py / visualize.py

Useful outputs:

# A. cumulative 2-D heatmap after summing over lag
#       x-axis: temperature
#       y-axis: humidity
#       colour: RR/OR
#
# B. temperature-response curves conditional on selected humidity values
#       e.g. 20th / 50th / 80th / 95th percentile humidity
#
# C. lag-specific temperature x humidity heatmaps
#       lag 0, 1, 3, 7, 14, 21
#
# D. optional interaction contrast heatmap
#       log RR_joint - log RR_tas - log RR_hurs

The primary publication plot would usually be the cumulative temperature x
humidity heatmap/contour because a full (temperature, humidity, lag, RR) object
has four displayed dimensions and is difficult to read in one figure.
"""

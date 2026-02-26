# models.py

import torch
import torch.nn as nn


class ExULayer(nn.Module):
    def __init__(self, in_features, out_features, weight_mean=3.5):
        super().__init__()
        self.weights = nn.Parameter(
            torch.normal(mean=weight_mean, std=0.5, size=(in_features, out_features))
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_features).uniform_(0, 1))

    def forward(self, x):
        x_shifted = x.unsqueeze(-1) - self.bias
        return torch.sum(x_shifted * torch.exp(self.weights), dim=1)


class SurfaceExUEncoder(nn.Module):
    def __init__(self, out_features, activation_fn, exu_mean_val=3.5,
                 use_exu_lag=False, exu_mean_lag=3.5):
        super().__init__()
        # FIX: Guard against odd out_features to avoid silently dropping a unit
        half_out = out_features // 2
        remainder = out_features - half_out  # val path gets the extra unit if odd
        self.act = activation_fn()

        # Exposure value path (ExU)
        self.exu_val = ExULayer(1, remainder, weight_mean=exu_mean_val)

        # Lag path: toggleable between ExU and Linear
        self.use_exu_lag = use_exu_lag
        if use_exu_lag:
            self.lag_layer = ExULayer(1, half_out, weight_mean=exu_mean_lag)
        else:
            self.lag_layer = nn.Linear(1, half_out)

    def forward(self, x):
        val_feat = self.act(self.exu_val(x[:, 0:1]))

        if self.use_exu_lag:
            lag_feat = self.act(self.lag_layer(x[:, 1:2]))
        else:
            # Sigmoid for the linear lag path for stable bounded outputs
            lag_feat = torch.sigmoid(self.lag_layer(x[:, 1:2]))

        return torch.cat([val_feat, lag_feat], dim=1)


class Multilayer_DLNAM(nn.Module):
    def __init__(self, exposure_dims, conf_dim, surface_configs, conf_configs,
                 num_subnets=3, trend_layers=None,
                 use_exu_exposure=True, use_exu_confounders=False, use_exu_trend=True,
                 use_exu_lag=False,
                 exu_mean_val=3.5, exu_mean_trend=3.5, exu_mean_lag=3.5,
                 surface_activation=nn.Mish, trend_activation=nn.Mish,
                 conf_activation=nn.Mish,
                 dropout_p=0.0,
                 subnet_dropout_p=0.0,
                 **kwargs):
        """
        dropout_p        : standard unit dropout applied *inside* each subnet's
                           hidden layers.  Regularises individual neurons.
                           Maps to the 'dropout' regularisation in the NAM paper.

        subnet_dropout_p : feature/subnet dropout as described in the NAM paper.
                           During training, each subnet's entire output contribution
                           is zeroed independently with this probability, then
                           rescaled so the expected sum is preserved.  This prevents
                           correlated features from arbitrarily splitting their
                           effect across subnets, improving additive identifiability.
                           Maps directly to 'feature dropout' in the NAM paper.
                           Set to 0.0 to disable (default).
        """
        super(Multilayer_DLNAM, self).__init__()

        if trend_layers is None:
            trend_layers = [128, 128, 64]

        self.bias = nn.Parameter(torch.zeros(1))
        self.exposure_lags = exposure_dims
        self.subnet_dropout_p = subnet_dropout_p

        # 1. Surfaces (one per exposure variable)
        self.surface_subnets = nn.ModuleList([
            nn.ModuleList([
                self._build_net(
                    2, 1, layers, surface_activation, dropout_p,
                    use_exu=use_exu_exposure, is_surface=True,
                    exu_mean_val=exu_mean_val,
                    use_exu_lag=use_exu_lag,
                    exu_mean_lag=exu_mean_lag
                )
                for _ in range(num_subnets)
            ]) for layers in surface_configs
        ])

        # 2. Confounders (one ModuleList of subnets per confounder)
        self.conf_subnets = nn.ModuleList([
            nn.ModuleList([
                self._build_net(
                    1, 1, layers, conf_activation, dropout_p,
                    use_exu=use_exu_confounders,
                    exu_mean_val=exu_mean_val
                )
                for _ in range(num_subnets)
            ]) for layers in conf_configs
        ])

        # 3. Trend (single set of subnets shared across time)
        self.trend_subnets = nn.ModuleList([
            self._build_net(
                1, 1, trend_layers, trend_activation, dropout_p,
                use_exu=use_exu_trend, exu_mean_val=exu_mean_trend
            )
            for _ in range(num_subnets)
        ])

        # Learnable ensemble mixing weights (softmax-normalised in forward)
        self.exp_weights = nn.Parameter(
            torch.randn(len(surface_configs), num_subnets) * 0.1
        )
        self.conf_weights = nn.Parameter(
            torch.randn(conf_dim, num_subnets) * 0.1
        )
        self.trend_weights = nn.Parameter(
            torch.randn(1, num_subnets) * 0.1
        )

    # ------------------------------------------------------------------
    # Network builder
    # ------------------------------------------------------------------
    def _build_net(self, in_dim, out_dim, layers, activation, dropout_p,
                   use_exu=False, is_surface=False, exu_mean_val=3.5,
                   use_exu_lag=False, exu_mean_lag=3.5):
        net = []

        if use_exu and is_surface:
            # SurfaceExUEncoder handles the first layer for 2-D surface inputs
            net.append(
                SurfaceExUEncoder(
                    layers[0], activation,
                    exu_mean_val, use_exu_lag, exu_mean_lag
                )
            )
            last_dim = layers[0]
            start_idx = 1
        elif use_exu:
            net.append(ExULayer(in_dim, layers[0], weight_mean=exu_mean_val))
            net.append(activation())
            last_dim = layers[0]
            start_idx = 1
        else:
            last_dim = in_dim
            start_idx = 0

        for i in range(start_idx, len(layers)):
            net.append(nn.Linear(last_dim, layers[i]))
            net.append(activation())
            if dropout_p > 0:
                net.append(nn.Dropout(p=dropout_p))
            last_dim = layers[i]

        net.append(nn.Linear(last_dim, out_dim))
        return nn.Sequential(*net)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x_exposures, x_conf, x_time):
        batch_size = x_time.shape[0]
        device = x_time.device
        log_mu = self.bias + torch.zeros((batch_size, 1), device=device)

        # ------------------------------------------------------------------
        # Subnet dropout mask helper (NAM paper: "feature dropout").
        # During training, independently zeros each subnet's entire output
        # with probability subnet_dropout_p, then rescales to preserve the
        # expected sum.  At eval time always returns all-ones (no dropout).
        # ------------------------------------------------------------------
        def _subnet_mask(n_subnets):
            if not self.training or self.subnet_dropout_p == 0.0:
                return None  # None signals: no masking needed
            keep_p = 1.0 - self.subnet_dropout_p
            mask = torch.bernoulli(
                torch.full((n_subnets,), keep_p, device=device)
            )
            scale = 1.0 / keep_p if keep_p > 0.0 else 0.0
            return mask * scale

        def _weighted_sum(subnets, inp, w_ens, mask):
            """
            Vectorised subnet aggregation using torch.stack.

            Instead of accumulating subnet outputs one at a time in a Python
            loop, all subnets are evaluated and their outputs stacked into a
            single tensor before the weighted sum.  This allows PyTorch to
            schedule the subnet forward passes more efficiently and performs
            the reduction in a single fused operation.

            subnets : list of nn.Module
            inp     : input tensor shared by all subnets
            w_ens   : (num_subnets,) softmax weights
            mask    : (num_subnets,) dropout mask or None
            """
            # Stack: (num_subnets, N, 1)
            stacked = torch.stack([s(inp) for s in subnets], dim=0)
            weights = w_ens  # (num_subnets,)
            if mask is not None:
                weights = weights * mask
            # Broadcast weights over (N, 1) dims → (num_subnets, 1, 1)
            return (weights.view(-1, 1, 1) * stacked).sum(dim=0)  # (N, 1)

        # --- Exposure surfaces ---
        for i, subnets in enumerate(self.surface_subnets):
            x_m = x_exposures[i]                           # (B, L+1)
            curr_lags_count = x_m.shape[1]

            lag_grid = _make_lag_grid(curr_lags_count, device)  # (L+1,)

            v_expanded = x_m.reshape(-1, 1)                     # (B*(L+1), 1)
            l_expanded = lag_grid.repeat(batch_size).unsqueeze(1)
            surf_input = torch.cat([v_expanded, l_expanded], dim=1)
            # (B*(L+1), 2)

            w_ens  = torch.softmax(self.exp_weights[i], dim=0)
            s_mask = _subnet_mask(len(subnets))

            # (B*(L+1), 1) → (B, L+1) → sum over lags → (B, 1)
            feat_flat = _weighted_sum(subnets, surf_input, w_ens, s_mask)
            log_mu = log_mu + feat_flat.view(batch_size, curr_lags_count).sum(
                dim=1, keepdim=True
            )

        # --- Confounders ---
        for k, subnets in enumerate(self.conf_subnets):
            val_k   = x_conf[:, k:k + 1]
            w_ens_k = torch.softmax(self.conf_weights[k], dim=0)
            s_mask  = _subnet_mask(len(subnets))
            log_mu  = log_mu + _weighted_sum(subnets, val_k, w_ens_k, s_mask)

        # --- Long-term trend ---
        w_ens_t    = torch.softmax(self.trend_weights[0], dim=0)
        s_mask     = _subnet_mask(len(self.trend_subnets))
        trend_eff  = _weighted_sum(self.trend_subnets, x_time, w_ens_t, s_mask)

        # Clamp both ends to avoid exp overflow / extreme underflow
        return torch.exp(torch.clamp(log_mu + trend_eff, min=-10.0, max=10.0))

    # ------------------------------------------------------------------
    # Interpretability helper – log-RR for a given surface
    # ------------------------------------------------------------------
    def get_log_rr(self, val_scaled, lag_scaled, surface_idx=0):
        """
        val_scaled : (N, 1) tensor of scaled exposure values
        lag_scaled : (N, 1) tensor of normalised lag values in [0, 1]

        Uses the same vectorised subnet aggregation as forward() so results
        are numerically identical to training-time predictions.
        """
        inputs  = torch.cat([val_scaled, lag_scaled], dim=1)
        subnets = self.surface_subnets[surface_idx]
        w_ens   = torch.softmax(self.exp_weights[surface_idx], dim=0)
        stacked = torch.stack([s(inputs) for s in subnets], dim=0)
        return (w_ens.view(-1, 1, 1) * stacked).sum(dim=0)  # (N, 1)


# ------------------------------------------------------------------
# Module-level helper so forward() and visualisation use the same grid
# ------------------------------------------------------------------
def _make_lag_grid(num_lags: int, device: torch.device) -> torch.Tensor:
    """
    Returns a 1-D tensor of length `num_lags` with values uniformly
    spaced in [0, 1].  This is the canonical lag normalisation used
    everywhere (forward pass AND visualisation).
    """
    return torch.linspace(0.0, 1.0, num_lags, device=device)
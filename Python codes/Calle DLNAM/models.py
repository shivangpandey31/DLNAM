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
        half_out  = out_features // 2
        remainder = out_features - half_out
        self.act  = activation_fn()

        self.exu_val     = ExULayer(1, remainder, weight_mean=exu_mean_val)
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
            lag_feat = torch.sigmoid(self.lag_layer(x[:, 1:2]))
        return torch.cat([val_feat, lag_feat], dim=1)


class CategoricalEncoding(nn.Module):
    """
    Categorical encoding as a proper neural network on the one-hot input.

    The input layer is W * 1(x) + b — a standard linear layer applied to
    the one-hot indicator vector 1(x) in {0,1}^C.  This is identical to
    an embedding lookup (selecting column x of W) but makes the NN
    interpretation explicit: it is the same first-layer structure used for
    all continuous inputs, just with a categorical (one-hot) input instead
    of a scalar.

    If hidden_layers is empty (default), the network reduces to a single
    linear layer W*1(x) + b -> scalar, which is exactly the lookup-table /
    treatment-contrast parameterisation.

    If hidden_layers is non-empty (e.g. [16]), one or more hidden layers
    with the given activation are appended after the input layer, giving
    the encoding additional expressive capacity.

    Parameters
    ----------
    num_categories : int
        Number of distinct category levels C.
    hidden_layers  : list of int
        Widths of optional hidden layers after the input layer.
        [] (default) reduces to a linear lookup table.
    activation     : nn.Module class
        Activation used in hidden layers (ignored if hidden_layers=[]).
    name           : str
        Human-readable label for diagnostics.
    """
    def __init__(self, num_categories: int,
                 hidden_layers=None,
                 activation=nn.Mish,
                 name: str = 'encoding'):
        super().__init__()
        self.name           = name
        self.num_categories = num_categories

        if hidden_layers is None:
            hidden_layers = []

        # Input layer: linear map from one-hot (C,) to first width (or 1)
        first_out = hidden_layers[0] if hidden_layers else 1
        net       = [nn.Linear(num_categories, first_out)]

        # Optional hidden layers
        last_dim  = first_out
        for width in hidden_layers[1:]:
            net.append(activation())
            net.append(nn.Linear(last_dim, width))
            last_dim = width

        # Activation after input layer only if hidden layers follow
        if hidden_layers:
            net.insert(1, activation())   # activate after input layer
            net.append(activation())      # activate before final layer
            net.append(nn.Linear(last_dim, 1))

        self.net = nn.Sequential(*net)

        # Zero-initialise the input layer weights and biases so the
        # encoding starts with no effect — identical to zero-init embeddings
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B,) LongTensor of category indices in {0,...,C-1}
        # One-hot encode then pass through network
        one_hot = torch.zeros(x.shape[0], self.num_categories,
                              device=x.device)
        one_hot.scatter_(1, x.long().unsqueeze(1), 1.0)
        return self.net(one_hot)   # (B, 1)


class Multilayer_DLNAM(nn.Module):
    def __init__(self,
                 # --- data dimensions ---
                 exposure_dims,
                 conf_dim,
                 # --- network configs ---
                 surface_configs,
                 conf_configs,
                 trend_layers=None,
                 # --- subnets ---
                 num_subnets=3,
                 # --- activations ---
                 surface_activation=nn.Mish,
                 trend_activation=nn.Mish,
                 conf_activation=nn.Mish,
                 # --- ExU ---
                 use_exu_exposure=True,
                 use_exu_confounders=False,
                 use_exu_trend=True,
                 use_exu_lag=False,
                 exu_mean_val=3.5,
                 exu_mean_trend=3.5,
                 exu_mean_lag=3.5,
                 # --- regularisation ---
                 dropout_p=0.0,
                 subnet_dropout_p=0.0,
                 # --- mixing weights ---
                 constrain_weights=True,
                 # --- categorical encodings ---
                 encoding_configs=None,
                 **kwargs):
        """
        encoding_configs : list of dicts or None
            Each dict must contain:
              'name'           : str       — label for diagnostics
              'num_categories' : int       — number of distinct levels
            Optional keys:
              'hidden_layers'  : list[int] — widths of hidden layers after the
                                             one-hot input layer. Default [] gives
                                             a pure lookup table (linear regression
                                             on one-hot input, no activation).
              'activation'     : nn.Module — activation for hidden layers.
                                             Default nn.Mish.

        The encoding modules are instantiated AFTER all subnet and mixing weight
        parameters so that the random state at surface/conf/trend initialisation
        is identical to a model with no encodings, ensuring reproducibility when
        comparing models with and without categorical effects.
        """
        super(Multilayer_DLNAM, self).__init__()

        if trend_layers    is None: trend_layers    = [128, 128, 64]
        if encoding_configs is None: encoding_configs = []

        self.bias              = nn.Parameter(torch.zeros(1))
        self.exposure_lags     = exposure_dims
        self.subnet_dropout_p  = subnet_dropout_p
        self.constrain_weights = constrain_weights

        # ------------------------------------------------------------------
        # 1. Exposure surfaces
        # ------------------------------------------------------------------
        self.surface_subnets = nn.ModuleList([
            nn.ModuleList([
                self._build_net(
                    2, 1, layers, surface_activation, dropout_p,
                    use_exu=use_exu_exposure, is_surface=True,
                    exu_mean_val=exu_mean_val,
                    use_exu_lag=use_exu_lag,
                    exu_mean_lag=exu_mean_lag,
                )
                for _ in range(num_subnets)
            ]) for layers in surface_configs
        ])

        # ------------------------------------------------------------------
        # 2. Confounders
        # ------------------------------------------------------------------
        self.conf_subnets = nn.ModuleList([
            nn.ModuleList([
                self._build_net(
                    1, 1, layers, conf_activation, dropout_p,
                    use_exu=use_exu_confounders,
                    exu_mean_val=exu_mean_val,
                )
                for _ in range(num_subnets)
            ]) for layers in conf_configs
        ])

        # ------------------------------------------------------------------
        # 3. Trend
        # ------------------------------------------------------------------
        self.trend_subnets = nn.ModuleList([
            self._build_net(
                1, 1, trend_layers, trend_activation, dropout_p,
                use_exu=use_exu_trend, exu_mean_val=exu_mean_trend,
            )
            for _ in range(num_subnets)
        ])

        # ------------------------------------------------------------------
        # 4. Mixing weights
        # ------------------------------------------------------------------
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
        # 5. Categorical encodings — instantiated LAST so that the random
        #    state during surface/conf/trend init is unaffected by whether
        #    encodings are present, preserving reproducibility across runs
        #    with and without categorical variables.
        # ------------------------------------------------------------------
        self.encodings = nn.ModuleList([
            CategoricalEncoding(
                num_categories = ec['num_categories'],
                hidden_layers  = ec.get('hidden_layers', []),
                activation     = ec.get('activation', nn.Mish),
                name           = ec.get('name', f'enc_{i}'),
            )
            for i, ec in enumerate(encoding_configs)
        ])

    # ------------------------------------------------------------------
    # Network builder
    # ------------------------------------------------------------------
    def _build_net(self, in_dim, out_dim, layers, activation, dropout_p,
                   use_exu=False, is_surface=False, exu_mean_val=3.5,
                   use_exu_lag=False, exu_mean_lag=3.5):
        net = []

        if use_exu and is_surface:
            net.append(SurfaceExUEncoder(
                layers[0], activation, exu_mean_val, use_exu_lag, exu_mean_lag
            ))
            last_dim  = layers[0]
            start_idx = 1
        elif use_exu:
            net.append(ExULayer(in_dim, layers[0], weight_mean=exu_mean_val))
            net.append(activation())
            last_dim  = layers[0]
            start_idx = 1
        else:
            last_dim  = in_dim
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
    # Mixing weight helper
    # ------------------------------------------------------------------
    def _mix_weights(self, raw):
        if self.constrain_weights:
            return torch.softmax(raw, dim=0)
        return raw

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x_exposures, x_conf, x_time, x_encodings=None):
        """
        x_encodings : list of LongTensors, one per entry in encoding_configs,
                      or None.
        """
        batch_size = x_time.shape[0]
        device     = x_time.device
        log_mu     = self.bias + torch.zeros((batch_size, 1), device=device)

        def _subnet_mask(n_subnets):
            if not self.training or self.subnet_dropout_p == 0.0:
                return None
            keep_p = 1.0 - self.subnet_dropout_p
            mask   = torch.bernoulli(
                torch.full((n_subnets,), keep_p, device=device)
            )
            return mask * (1.0 / keep_p if keep_p > 0.0 else 0.0)

        def _weighted_sum(subnets, inp, w_ens, mask):
            stacked = torch.stack([s(inp) for s in subnets], dim=0)
            weights = w_ens if mask is None else w_ens * mask
            return (weights.view(-1, 1, 1) * stacked).sum(dim=0)

        # --- Exposure surfaces ---
        for i, subnets in enumerate(self.surface_subnets):
            x_m             = x_exposures[i]
            curr_lags_count = x_m.shape[1]
            lag_grid        = _make_lag_grid(curr_lags_count, device)
            v_expanded      = x_m.reshape(-1, 1)
            l_expanded      = lag_grid.repeat(batch_size).unsqueeze(1)
            surf_input      = torch.cat([v_expanded, l_expanded], dim=1)
            w_ens           = self._mix_weights(self.exp_weights[i])
            s_mask          = _subnet_mask(len(subnets))
            feat_flat       = _weighted_sum(subnets, surf_input, w_ens, s_mask)
            log_mu = log_mu + feat_flat.view(batch_size, curr_lags_count).sum(
                dim=1, keepdim=True
            )

        # --- Confounders ---
        for k, subnets in enumerate(self.conf_subnets):
            val_k   = x_conf[:, k:k + 1]
            w_ens_k = self._mix_weights(self.conf_weights[k])
            s_mask  = _subnet_mask(len(subnets))
            log_mu  = log_mu + _weighted_sum(subnets, val_k, w_ens_k, s_mask)

        # --- Trend ---
        w_ens_t = self._mix_weights(self.trend_weights[0])
        s_mask  = _subnet_mask(len(self.trend_subnets))
        log_mu  = log_mu + _weighted_sum(
            self.trend_subnets, x_time, w_ens_t, s_mask
        )

        # --- Categorical encodings ---
        if x_encodings is not None:
            for enc_module, x_enc in zip(self.encodings, x_encodings):
                log_mu = log_mu + enc_module(x_enc)

        return torch.exp(torch.clamp(log_mu, min=-10.0, max=10.0))

    # ------------------------------------------------------------------
    # Interpretability helper
    # ------------------------------------------------------------------
    def get_log_rr(self, val_scaled, lag_scaled, surface_idx=0):
        inputs  = torch.cat([val_scaled, lag_scaled], dim=1)
        subnets = self.surface_subnets[surface_idx]
        w_ens   = self._mix_weights(self.exp_weights[surface_idx])
        stacked = torch.stack([s(inputs) for s in subnets], dim=0)
        return (w_ens.view(-1, 1, 1) * stacked).sum(dim=0)


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------
def _make_lag_grid(num_lags: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_lags, device=device)
"""
laplace.py -- model-level last-layer Laplace confidence intervals.

The interval is built from the joint Fisher information of the fitted DLNAM on
the link scale. Conditional on the learned hidden features and subnet-mixing
weights, the collected parameters are the global intercept and every term's final
linear layer. The covariance is therefore joint across all additive terms, so
multi-exposure models and smooth covariates are handled by the same code path.

The reported interval is for a centered term effect, e.g. a cumulative
exposure-response curve for a surface term, the pointwise value-by-lag surface,
or a one-dimensional covariate shape function. The intercept is included in the
fitted covariance, but its derivative is zero for centered contrasts; this
mirrors standard GLM/DLNM inference where the intercept is part of the covariance
fit even when the reported contrast does not include an intercept column.
"""
from __future__ import annotations

import numpy as np
import torch


def _final_linear(module):
    """Return the last nn.Linear in a term/subnet, if there is one."""
    if isinstance(module, torch.nn.Sequential):
        if len(module) and isinstance(module[-1], torch.nn.Linear):
            return module[-1]
        return None
    if isinstance(module, torch.nn.Linear):
        return module
    for attr in ("tail", "net", "layers"):
        seq = getattr(module, attr, None)
        if isinstance(seq, torch.nn.Linear):
            return seq
        if (isinstance(seq, torch.nn.Sequential) and len(seq)
                and isinstance(seq[-1], torch.nn.Linear)):
            return seq[-1]
    return None


def _term_last_layer_params(term):
    """Final linear weight/bias tensors for a term."""
    emb = getattr(term, "emb", None)
    tail = getattr(term, "tail", None)
    if emb is not None and isinstance(tail, torch.nn.Identity):
        # A pure categorical term is already a linear set of level effects, so
        # the embedding weights are its analogue of a final coefficient vector.
        return [emb.weight]

    params = []
    subnets = getattr(term, "subnets", None)
    if subnets is not None:
        modules = list(subnets)
    else:
        modules = [term]
    for module in modules:
        lin = _final_linear(module)
        if lin is not None:
            params.extend([lin.weight, lin.bias])
    return params


def collect_last_layer_params(model):
    """Ordered `(name, parameter)` list used by the Laplace approximation.

    Included by design:
      * global intercept;
      * final linear layers of every additive term/subnet.

    The intercept is not prior-penalised in the covariance. The final-layer
    parameters share the evidence-selected prior precision. Subnet mixing
    weights are treated as fixed at their fitted values.
    """
    out = [("intercept", model.intercept)]
    for term_name, term in model.terms.items():
        for i, param in enumerate(_term_last_layer_params(term)):
            out.append((f"{term_name}.ll{i}", param))
    return out


def _jacobian_rows(vector_fn, params, n_out):
    """Dense Jacobian of `vector_fn() -> (n_out,)` w.r.t. `params`.

    This uses one reverse-mode pass per output entry. It is slower than a
    hand-coded design matrix, but robust across surfaces, smooth terms, trend
    terms, and future term implementations because it differentiates the actual
    model path used for training and prediction.
    """
    out = vector_fn()
    P = int(sum(p.numel() for p in params))
    J = np.zeros((n_out, P), dtype=float)
    for k in range(n_out):
        grads = torch.autograd.grad(out[k], params, retain_graph=True,
                                    allow_unused=True)
        row = []
        for grad, param in zip(grads, params):
            row.append((torch.zeros_like(param) if grad is None else grad).reshape(-1))
        J[k] = torch.cat(row).detach().cpu().numpy()
    return J


def pooled_evidence_lambda(evidence_terms, iters=300, tol=1e-7):
    """Shared prior precision for an ensemble via pooled MacKay iteration.

    Each member contributes eigenvalues of its likelihood information restricted
    to the prior-penalised parameters and the squared norm of those parameters.
    The global intercept is deliberately excluded from this prior fit.
    """
    terms = [(np.asarray(evals, float).clip(0.0), max(float(theta_sq), 1e-12))
             for evals, theta_sq in evidence_terms]
    theta_sq_sum = sum(theta_sq for _, theta_sq in terms)
    if theta_sq_sum <= 0:
        return 1.0

    lam = 1.0
    for _ in range(iters):
        gamma = sum(float(np.sum(evals / (evals + lam))) for evals, _ in terms)
        new = min(max(gamma / theta_sq_sum, 1e-8), 1e12)
        if abs(np.log(new) - np.log(lam)) < tol:
            lam = new
            break
        lam = new
    return float(lam)


class LastLayerLaplace:
    """Joint last-layer Laplace approximation for one DLNAM member."""

    def __init__(self, model, prepared_inputs, mu_hat, *, phi=1.0,
                 ridge=1e-6, prior_precision=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.inputs = {k: v.to(self.device) for k, v in prepared_inputs.items()}
        self.mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1).clip(1e-9)
        self.phi = float(phi)
        self.ridge = float(ridge)
        self.named_params = collect_last_layer_params(model)
        self.params = [p for _, p in self.named_params]
        self.P = int(sum(p.numel() for p in self.params))
        self.prior_mask = self._prior_mask()
        self._FtWF = None
        self._eigcache = None
        self._prior_precision = prior_precision

    def _prior_mask(self):
        mask = []
        for name, param in self.named_params:
            value = 0.0 if name == "intercept" else 1.0
            mask.extend([value] * param.numel())
        return np.asarray(mask, dtype=float)

    def _N(self):
        return next(iter(self.inputs.values())).shape[0]

    def _eta_vector(self):
        eta = self.model.intercept.expand(self._N())
        for name, term in self.model.terms.items():
            eta = eta + term(self.inputs[name]).reshape(-1)
        return eta

    def _ftwf(self):
        """Poisson/quasi-Poisson last-layer information Phi^T W Phi."""
        if self._FtWF is None:
            Phi = _jacobian_rows(self._eta_vector, self.params, self._N())
            if len(self.mu_hat) != Phi.shape[0]:
                raise ValueError("mu_hat length does not match prepared input rows")
            self._FtWF = (Phi * self.mu_hat[:, None]).T @ Phi
        return self._FtWF

    def evidence_terms(self):
        """Return sufficient statistics for shared lambda selection."""
        penalized = self.prior_mask > 0
        if not np.any(penalized):
            return np.zeros(0), 0.0
        F = self._ftwf() / max(self.phi, 1e-12)
        M = F[np.ix_(penalized, penalized)]
        unpenalized = ~penalized
        if np.any(unpenalized):
            F_pu = F[np.ix_(penalized, unpenalized)]
            F_uu = F[np.ix_(unpenalized, unpenalized)]
            M = M - F_pu @ np.linalg.pinv(F_uu, rcond=1e-12) @ F_pu.T
        M = 0.5 * (M + M.T)
        evals = np.linalg.eigvalsh(M).clip(0.0)
        theta = np.concatenate([
            p.detach().cpu().numpy().reshape(-1) for p in self.params
        ])[penalized]
        return evals, float(theta @ theta)

    def set_prior_precision(self, value):
        self._prior_precision = float(value)
        self._eigcache = None

    @property
    def prior_precision(self):
        if self._prior_precision is None:
            self._prior_precision = pooled_evidence_lambda([self.evidence_terms()])
        return self._prior_precision

    def _eigh_hessian(self):
        """Eigenbasis of the penalised Hessian used for covariance propagation."""
        if self._eigcache is None:
            H = (self._ftwf()
                 + self.phi * self.prior_precision * np.diag(self.prior_mask)
                 + self.ridge * np.eye(self.P))
            w, V = np.linalg.eigh(H)
            self._eigcache = (w.clip(1e-30), V)
        return self._eigcache

    def _term_values(self, term, raw_values):
        raw = np.asarray(raw_values, dtype=float)
        if hasattr(term, "lag_grid"):
            scaled = term._to_scaled(raw)
            x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            win = x.view(-1, 1).repeat(1, int(term.lag_max) + 1)
            return term(win).reshape(-1)

        if hasattr(term, "_to_scaled"):
            scaled = term._to_scaled(raw)
            x = torch.tensor(scaled, dtype=torch.float32, device=self.device).view(-1, 1)
            return term(x).reshape(-1)

        if hasattr(term, "num_categories"):
            idx = torch.tensor(np.rint(raw).astype(int), dtype=torch.long,
                               device=self.device)
            return term(idx).reshape(-1)

        raise NotImplementedError(
            f"Laplace effect SE is not implemented for term type {type(term).__name__}"
        )

    def _surface_values(self, term, grid_raw, ref_raw):
        """Pointwise surface contrast f(x, lag) - f(ref, lag), lag-major."""
        if not hasattr(term, "lag_grid"):
            raise ValueError("surface_se is only defined for surface terms")
        grid_raw = np.asarray(grid_raw, dtype=float)
        scaled = term._to_scaled(grid_raw)
        ref_scaled = float(term._to_scaled(np.asarray([ref_raw], dtype=float))[0])
        G = len(scaled)
        lags = term.lag_grid.to(self.device)
        Lp1 = int(lags.numel())
        vv = torch.tensor(np.tile(scaled, Lp1), dtype=torch.float32,
                          device=self.device).view(-1, 1)
        rr = torch.full((Lp1 * G, 1), ref_scaled, dtype=torch.float32,
                        device=self.device)
        ll = lags.repeat_interleave(G).view(-1, 1)
        ev = term._mixed(torch.cat([vv, ll], dim=1)).reshape(-1)
        er = term._mixed(torch.cat([rr, ll], dim=1)).reshape(-1)
        return ev - er

    def _effect_A(self, term_name, grid_raw, ref_raw):
        term = self.model.term(term_name)
        grid_raw = np.asarray(grid_raw, dtype=float)
        ref_raw = np.asarray([ref_raw], dtype=float)

        def centered_effect():
            return self._term_values(term, grid_raw) - self._term_values(term, ref_raw)[0]

        return _jacobian_rows(centered_effect, self.params, len(grid_raw))

    def _se_from_design(self, A):
        w, V = self._eigh_hessian()
        AV = A @ V
        var = (AV ** 2) @ (self.phi / w)
        return np.sqrt(var.clip(0.0))

    def effect_se(self, term_name, grid_raw, ref_raw):
        """SE of a centered term effect on the log/link scale."""
        return self._se_from_design(self._effect_A(term_name, grid_raw, ref_raw))

    def surface_se(self, term_name, grid_raw, ref_raw):
        """SE for f(x, lag) - f(ref, lag), returned as (n_lags, n_grid)."""
        term = self.model.term(term_name)
        grid_raw = np.asarray(grid_raw, dtype=float)
        G = len(grid_raw)
        Lp1 = int(term.lag_grid.numel())

        def centered_surface():
            return self._surface_values(term, grid_raw, ref_raw)

        A = _jacobian_rows(centered_surface, self.params, Lp1 * G)
        return self._se_from_design(A).reshape(Lp1, G)


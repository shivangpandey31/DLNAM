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


def collect_last_layer_params(model, include_terms=None):
    """Ordered `(name, parameter)` list used by the Laplace approximation.

    Included by design:
      * global intercept;
      * final linear layers of every included additive term/subnet.

    The intercept is not prior-penalised in the covariance. The final-layer
    parameters share the evidence-selected prior precision. Subnet mixing
    weights are treated as fixed at their fitted values.
    """
    include_terms = None if include_terms is None else set(include_terms)
    out = [("intercept", model.intercept)]
    for term_name, term in model.terms.items():
        if include_terms is not None and term_name not in include_terms:
            continue
        for i, param in enumerate(_term_last_layer_params(term)):
            out.append((f"{term_name}.ll{i}", param))
    return out


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

    def __init__(self, model, prepared_inputs, information_weight=None, *, phi=1.0,
                 ridge=1e-6, prior_precision=None, include_terms=None,
                 fisher_batch_size=4096, eliminate_index=None,
                 profile_probability=None, profile_event_total=None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        selected = (set(model.terms) if include_terms is None
                    else set(include_terms))
        self.inputs = {
            k: prepared_inputs[k].to(self.device) for k in selected
        }
        self.information_weight = (
            None if information_weight is None else
            np.asarray(information_weight, dtype=float).reshape(-1).clip(1e-12)
        )
        self.eliminate_index = (
            None if eliminate_index is None else
            np.asarray(eliminate_index, dtype=np.int64).reshape(-1)
        )
        self.profile_probability = (
            None if profile_probability is None else
            np.asarray(profile_probability, dtype=float).reshape(-1)
        )
        self.profile_event_total = (
            None if profile_event_total is None else
            np.asarray(profile_event_total, dtype=float).reshape(-1)
        )
        self.phi = float(phi)
        self.ridge = float(ridge)
        self.include_terms = None if include_terms is None else tuple(include_terms)
        self.fisher_batch_size = max(int(fisher_batch_size), 1)
        self.named_params = collect_last_layer_params(model, include_terms)
        self.params = [p for _, p in self.named_params]
        self.P = int(sum(p.numel() for p in self.params))
        self.prior_mask = self._prior_mask()
        self._FtWF = None
        self._eigcache = None
        self._prior_precision = prior_precision

        profiled_args = (
            self.eliminate_index is not None,
            self.profile_probability is not None,
            self.profile_event_total is not None,
        )
        if any(profiled_args) and not all(profiled_args):
            raise ValueError(
                "profiled Laplace requires eliminate_index, profile_probability, "
                "and profile_event_total together"
            )
        if not any(profiled_args) and self.information_weight is None:
            raise ValueError(
                "standard Laplace requires information_weight"
            )

    def _prior_mask(self):
        mask = []
        for name, param in self.named_params:
            value = 0.0 if name == "intercept" else 1.0
            mask.extend([value] * param.numel())
        return np.asarray(mask, dtype=float)

    def _N(self):
        return next(iter(self.inputs.values())).shape[0]

    def _design_block(self, start, stop):
        """Last-layer Jacobian rows for one observation block.

        Conditional on the fitted representation and mixing weights, these
        rows are the exact linear design. No numerical differentiation or
        approximation is involved.
        """
        B = stop - start
        blocks = [torch.ones((B, 1), device=self.device,
                             dtype=self.model.intercept.dtype)]
        for name, term in self.model.terms.items():
            if self.include_terms is not None and name not in self.include_terms:
                continue
            design_fn = getattr(term, "_last_layer_design", None)
            if design_fn is None:
                raise NotImplementedError(
                    f"last-layer design is unavailable for {type(term).__name__}"
                )
            blocks.append(design_fn(self.inputs[name][start:stop]))
        design = torch.cat(blocks, dim=1)
        if design.shape != (B, self.P):
            raise RuntimeError(
                f"last-layer design has shape {tuple(design.shape)}, "
                f"expected {(B, self.P)}"
            )
        return design

    def _design_indices(self, indices):
        """Last-layer design for arbitrary observation indices."""
        idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        B = int(idx.numel())
        blocks = [torch.ones((B, 1), device=self.device,
                             dtype=self.model.intercept.dtype)]
        for name, term in self.model.terms.items():
            if self.include_terms is not None and name not in self.include_terms:
                continue
            design_fn = getattr(term, "_last_layer_design", None)
            if design_fn is None:
                raise NotImplementedError(
                    f"last-layer design is unavailable for {type(term).__name__}"
                )
            blocks.append(design_fn(self.inputs[name][idx]))
        design = torch.cat(blocks, dim=1)
        if design.shape != (B, self.P):
            raise RuntimeError(
                f"last-layer design has shape {tuple(design.shape)}, "
                f"expected {(B, self.P)}"
            )
        return design

    def _ftwf(self):
        """Likelihood information for the fitted last-layer parameters.

        Standard GLMs use ``Phi^T W Phi``. For profiled Poisson strata, the
        nuisance intercept information is removed analytically and each stratum
        contributes ``Y_g * Cov_p(Phi_g)`` under its fitted within-stratum
        softmax probabilities.
        """
        if self._FtWF is None:
            N = self._N()
            if self.eliminate_index is not None:
                self._FtWF = self._profiled_information()
                return self._FtWF
            if len(self.information_weight) != N:
                raise ValueError("information weights do not match prepared input rows")
            F = np.zeros((self.P, self.P), dtype=float)
            batch_size = min(self.fisher_batch_size, N)
            start = 0
            with torch.no_grad():
                while start < N:
                    stop = min(start + batch_size, N)
                    try:
                        block = self._design_block(start, stop)
                    except torch.OutOfMemoryError:
                        if batch_size == 1:
                            raise
                        batch_size = max(batch_size // 2, 1)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    Phi = block.detach().cpu().numpy().astype(float, copy=False)
                    weight = self.information_weight[start:stop]
                    F += (Phi * weight[:, None]).T @ Phi
                    start = stop
            self.fisher_batch_size = batch_size
            self._FtWF = 0.5 * (F + F.T)
        return self._FtWF

    def _profiled_information(self):
        """Fisher information after profiling Poisson stratum intercepts."""
        N = self._N()
        g = self.eliminate_index
        prob = self.profile_probability
        if len(g) != N or len(prob) != N:
            raise ValueError(
                "profiled Laplace group/probability vectors do not match input rows"
            )
        if np.any(prob < 0) or not np.all(np.isfinite(prob)):
            raise ValueError("invalid profiled within-stratum probabilities")

        G = int(g.max()) + 1 if len(g) else 0
        if G <= 0 or len(self.profile_event_total) != G:
            raise ValueError("profile_event_total does not match elimination strata")

        counts = np.bincount(g, minlength=G)
        order = np.argsort(g, kind="stable")
        offsets = np.concatenate([[0], np.cumsum(counts)])

        F = np.zeros((self.P, self.P), dtype=float)
        # fisher_batch_size is interpreted as an approximate ROW budget. Build
        # each chunk from complete groups so the profiled covariance is exact.
        start_group = 0
        with torch.no_grad():
            while start_group < G:
                stop_group = start_group
                rows = 0
                while stop_group < G and (rows < self.fisher_batch_size or stop_group == start_group):
                    rows += int(counts[stop_group])
                    stop_group += 1

                row_idx = order[offsets[start_group]:offsets[stop_group]]
                try:
                    block = self._design_indices(row_idx)
                except torch.OutOfMemoryError:
                    if stop_group - start_group <= 1:
                        raise
                    # Reduce the future row budget and retry this same group range
                    # with fewer groups.
                    self.fisher_batch_size = max(self.fisher_batch_size // 2, 1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                Phi = block.detach().cpu().numpy().astype(float, copy=False)
                local_g = g[row_idx] - start_group
                p = prob[row_idx]
                ytot = self.profile_event_total[start_group:stop_group]

                # First term: sum_g Y_g sum_j p_gj x_gj x_gj^T
                row_weight = ytot[local_g] * p
                F += (Phi * row_weight[:, None]).T @ Phi

                # Second term: sum_g Y_g xbar_g xbar_g^T
                K = stop_group - start_group
                means = np.zeros((K, self.P), dtype=float)
                np.add.at(means, local_g, Phi * p[:, None])
                F -= (means * ytot[:, None]).T @ means

                start_group = stop_group

        return 0.5 * (F + F.T)

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

    def _term_columns(self, term_name):
        columns = []
        offset = 0
        prefix = f"{term_name}.ll"
        for name, param in self.named_params:
            indices = range(offset, offset + param.numel())
            if name.startswith(prefix):
                columns.extend(indices)
            offset += param.numel()
        return columns

    def _effect_A(self, term_name, grid_raw, ref_raw):
        term = self.model.term(term_name)
        grid_raw = np.asarray(grid_raw, dtype=float)
        raw = np.concatenate([grid_raw, np.asarray([ref_raw], dtype=float)])

        if hasattr(term, "lag_grid"):
            scaled = term._to_scaled(raw)
            x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            x = x.view(-1, 1).repeat(1, int(term.lag_max) + 1)
        elif hasattr(term, "_to_scaled"):
            scaled = term._to_scaled(raw)
            x = torch.tensor(scaled, dtype=torch.float32,
                             device=self.device).view(-1, 1)
        elif hasattr(term, "num_categories"):
            x = torch.tensor(np.rint(raw).astype(int), dtype=torch.long,
                             device=self.device)
        else:
            raise NotImplementedError(
                f"Laplace effect SE is unavailable for {type(term).__name__}"
            )

        with torch.no_grad():
            design = term._last_layer_design(x).detach().cpu().numpy()
        centered = design[:-1] - design[-1]

        columns = self._term_columns(term_name)
        if centered.shape[1] != len(columns):
            raise RuntimeError("term design does not match collected parameters")

        A = np.zeros((len(grid_raw), self.P), dtype=float)
        A[:, columns] = centered
        return A

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
        if not hasattr(term, "_last_layer_point_design"):
            raise ValueError("surface_se is only defined for surface terms")
        grid_raw = np.asarray(grid_raw, dtype=float)
        G = len(grid_raw)
        scaled = term._to_scaled(grid_raw)
        ref_scaled = float(term._to_scaled(np.asarray([ref_raw], dtype=float))[0])
        lags = term.lag_grid.to(self.device)
        Lp1 = int(lags.numel())
        vv = torch.tensor(np.tile(scaled, Lp1), dtype=torch.float32,
                          device=self.device).view(-1, 1)
        rr = torch.full((Lp1 * G, 1), ref_scaled, dtype=torch.float32,
                        device=self.device)
        ll = lags.repeat_interleave(G).view(-1, 1)
        with torch.no_grad():
            observed = term._last_layer_point_design(
                torch.cat([vv, ll], dim=1)
            ).detach().cpu().numpy()
            reference = term._last_layer_point_design(
                torch.cat([rr, ll], dim=1)
            ).detach().cpu().numpy()
        centered = observed - reference
        columns = self._term_columns(term_name)
        if centered.shape[1] != len(columns):
            raise RuntimeError("surface design does not match collected parameters")
        A = np.zeros((Lp1 * G, self.P), dtype=float)
        A[:, columns] = centered
        return self._se_from_design(A).reshape(Lp1, G)


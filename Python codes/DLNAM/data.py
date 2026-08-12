"""
data.py — DataFrame -> model-ready inputs.

V2 keeps the model itself data-source agnostic, while restoring the three useful
input modes from the original implementation:

``raw``
    One continuous time series. Surface lags are constructed globally.
``grouped``
    Repeated time series (person, region, etc.). Surface lags are constructed
    independently within each group, so histories never cross group boundaries.
``prelagged``
    Lag columns already exist as ``<surface_name>_lag0 ... _lagK`` and are read
    directly. This is useful for very large register/HPC pipelines.

Categorical/fixed-effect configuration lives in ModelConfig/CategoricalTermSpec;
DataProcessor only maps DataFrame levels to integer model indices. If a
CategoricalTermSpec has ``order=()``, the order is inferred from the data and
attached to every ensemble member. ``num_categories`` remains required because
it fixes the model parameter dimensions before DataProcessor runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch

from .config import (
    ModelConfig,
    SurfaceTermSpec,
    SmoothTermSpec,
    TrendTermSpec,
    CategoricalTermSpec,
)


InputType = Literal["raw", "grouped", "prelagged"]


def make_windows(series: np.ndarray, lag_max: int) -> np.ndarray:
    """(T,) -> (T-lag_max, lag_max+1), column j = lag j.

    Sample i corresponds to time i+lag_max, with column 0=current value and
    column j=value j steps earlier.
    """
    s = np.ascontiguousarray(series, dtype=np.float32)
    asc = np.lib.stride_tricks.sliding_window_view(s, lag_max + 1)
    return np.ascontiguousarray(asc[:, ::-1])


@dataclass
class PreparedData:
    inputs: dict
    y: torch.Tensor
    raw: dict
    n_samples: int
    row_index: Optional[np.ndarray] = None
    input_type: str = "raw"
    category_orders: Optional[dict] = None


class DataProcessor:
    def __init__(self, model_config: ModelConfig):
        self.cfg = model_config
        self.category_orders: dict[str, list] = {}

    @staticmethod
    def _ensemble_list(ensemble):
        # Accept either trainer.ensemble or the ensemble list itself.
        if hasattr(ensemble, "ensemble"):
            ensemble = ensemble.ensemble
        ensemble = list(ensemble)
        if not ensemble:
            raise ValueError("ensemble must contain at least one DLNAM model")
        return ensemble

    @staticmethod
    def _source_col(name: str, spec) -> str:
        if isinstance(spec, CategoricalTermSpec):
            return spec.source_col or name
        return name

    def _required_columns(self, target_col: str, input_type: InputType,
                          groupby_col: Optional[str], time_col: Optional[str]) -> list[str]:
        cols = [target_col]
        if input_type == "grouped" and groupby_col is not None:
            cols.append(groupby_col)
        if time_col is not None:
            cols.append(time_col)

        for name, spec in self.cfg.terms.items():
            if isinstance(spec, TrendTermSpec):
                continue
            if isinstance(spec, SurfaceTermSpec) and input_type == "prelagged":
                cols.extend(f"{name}_lag{i}" for i in range(spec.lag_max + 1))
            else:
                cols.append(self._source_col(name, spec))
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(cols))

    @staticmethod
    def _infer_order(values, expected_n: int, name: str) -> list:
        vals = list(pd.unique(pd.Series(values).dropna()))
        # Prefer a sorted order for reproducibility when categories are mutually
        # comparable; fall back to stable order-of-appearance for mixed types.
        try:
            vals = sorted(vals)
        except TypeError:
            pass
        if len(vals) != int(expected_n):
            raise ValueError(
                f"categorical term '{name}' expects num_categories={expected_n}, "
                f"but preparation data contains {len(vals)} distinct levels. "
                "Update num_categories or provide the intended order explicitly."
            )
        return vals

    def _category_order(self, name: str, spec: CategoricalTermSpec,
                        df: pd.DataFrame, ensemble, fit_metadata: bool = True) -> list:
        source = self._source_col(name, spec)
        if fit_metadata:
            order = list(spec.order) if spec.order else self._infer_order(
                df[source], spec.num_categories, name
            )
            self.category_orders[name] = order
        else:
            if name in self.category_orders:
                order = list(self.category_orders[name])
            elif spec.order:
                order = list(spec.order)
            else:
                raise ValueError(
                    f"categorical term '{name}' has no fitted category order. "
                    "Prepare training data first (fit_scaling=True) or supply spec.order."
                )
        if len(order) != int(spec.num_categories):
            raise ValueError(
                f"categorical term '{name}' order has {len(order)} levels but "
                f"num_categories={spec.num_categories}"
            )
        for m in ensemble:
            m.term(name).set_category_order(order)
        return order

    @staticmethod
    def _trend_values(df: pd.DataFrame, row_positions: np.ndarray,
                      time_col: Optional[str]) -> np.ndarray:
        n = len(row_positions)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if time_col is None:
            return np.linspace(0.0, 1.0, n, dtype=np.float32)

        s = df.iloc[row_positions][time_col]
        if pd.api.types.is_datetime64_any_dtype(s):
            vals = s.astype("int64").to_numpy(dtype=np.float64)
        elif pd.api.types.is_numeric_dtype(s):
            vals = s.to_numpy(dtype=np.float64)
        else:
            parsed = pd.to_datetime(s, errors="raise")
            vals = parsed.astype("int64").to_numpy(dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros(n, dtype=np.float32)
        return ((vals - lo) / (hi - lo)).astype(np.float32)

    def prepare(self, df: pd.DataFrame, ensemble, target_col: str,
                input_type: InputType = "raw",
                groupby_col: Optional[str] = None,
                time_col: Optional[str] = None,
                fit_scaling: bool = True) -> PreparedData:
        """Prepare model tensors and fit term scaling.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data. The DLNAM model never reads files itself.
        ensemble : sequence of DLNAM, or Trainer
            Scaling/category metadata are copied to every ensemble member.
        target_col : str
            Outcome column.
        input_type : {'raw', 'grouped', 'prelagged'}
            ``grouped`` restores the v1 behaviour of creating lag histories
            independently within each ``groupby_col``.
        groupby_col : str, optional
            Required for ``input_type='grouped'``.
        time_col : str, optional
            If supplied in grouped mode, rows are stably sorted by group+time
            before lag construction. It is also used to create a calendar-based
            normalised TrendTerm instead of a row-number trend.
        fit_scaling : bool, default True
            Fit exposure/confounder scaling and infer category order. Use True
            for training data. Reuse the SAME DataProcessor with False for
            validation/test data to prevent scaler/category leakage.

        Examples
        --------
        ``processor.prepare(df, ensemble, target_col='event',
                            input_type='grouped', groupby_col='id')``
        """
        if input_type not in ("raw", "grouped", "prelagged"):
            raise ValueError("input_type must be 'raw', 'grouped', or 'prelagged'")
        if input_type == "grouped" and not groupby_col:
            raise ValueError("input_type='grouped' requires groupby_col")

        ensemble = self._ensemble_list(ensemble)
        required = self._required_columns(target_col, input_type, groupby_col, time_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"missing required columns: {missing}")

        work = df.copy()
        work["__dlnam_original_index__"] = np.asarray(df.index)
        work = work.dropna(subset=required)
        if input_type == "grouped" and time_col is not None:
            work = work.sort_values([groupby_col, time_col], kind="mergesort")
        work = work.reset_index(drop=True)

        if input_type == "prelagged":
            return self._prepare_prelagged(work, ensemble, target_col, time_col, fit_scaling)
        if input_type == "grouped":
            return self._prepare_grouped(work, ensemble, target_col, groupby_col, time_col, fit_scaling)
        return self._prepare_raw(work, ensemble, target_col, time_col, fit_scaling)

    # ------------------------------------------------------------------
    # RAW: one global continuous time series
    # ------------------------------------------------------------------
    def _prepare_raw(self, df, ensemble, target_col, time_col, fit_scaling):
        surface_lags = [s.lag_max for s in self.cfg.terms.values()
                        if isinstance(s, SurfaceTermSpec)]
        total_lag = max(surface_lags) if surface_lags else 0
        n = len(df) - total_lag
        if n <= 0:
            raise ValueError("not enough rows after lag alignment")
        row_positions = np.arange(total_lag, len(df), dtype=int)
        inputs, raw = {}, {}

        for name, spec in self.cfg.terms.items():
            if isinstance(spec, SurfaceTermSpec):
                series = df[name].to_numpy(dtype=float)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(series)
                scaled = ensemble[0].term(name)._to_scaled(series)
                win = make_windows(scaled, spec.lag_max)
                win = win[total_lag - spec.lag_max:]
                inputs[name] = torch.from_numpy(win)
                raw[name] = series[row_positions]

            elif isinstance(spec, SmoothTermSpec):
                series = df[name].to_numpy(dtype=float)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(series)
                vals = ensemble[0].term(name)._to_scaled(series)[row_positions]
                inputs[name] = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
                raw[name] = series[row_positions]

            elif isinstance(spec, TrendTermSpec):
                for m in ensemble:
                    m.term(name).fit_scaling()
                vals = self._trend_values(df, row_positions, time_col)
                inputs[name] = torch.from_numpy(vals).view(-1, 1)
                raw[name] = vals

            elif isinstance(spec, CategoricalTermSpec):
                source = self._source_col(name, spec)
                order = self._category_order(name, spec, df, ensemble, fit_metadata=fit_scaling)
                enc = {cat: i for i, cat in enumerate(order)}
                mapped = df[source].map(enc).to_numpy()[row_positions]
                if pd.isna(mapped).any():
                    raise ValueError(f"unmapped category in '{source}'; check order")
                inputs[name] = torch.tensor(mapped.astype(np.int64), dtype=torch.long)
                raw[name] = df[source].to_numpy()[row_positions]

            else:
                raise NotImplementedError(f"no processor branch for {type(spec).__name__}")

        y = torch.tensor(df[target_col].to_numpy(dtype=float)[row_positions],
                         dtype=torch.float32).view(-1, 1)
        return self._pack(df, inputs, y, raw, row_positions, "raw")

    # ------------------------------------------------------------------
    # GROUPED: independent lag histories within each group
    # ------------------------------------------------------------------
    def _prepare_grouped(self, df, ensemble, target_col, groupby_col, time_col, fit_scaling):
        surface_lags = [s.lag_max for s in self.cfg.terms.values()
                        if isinstance(s, SurfaceTermSpec)]
        total_lag = max(surface_lags) if surface_lags else 0

        groups = []
        for group_value, positions in df.groupby(groupby_col, sort=False).indices.items():
            positions = np.asarray(positions, dtype=int)
            if len(positions) > total_lag:
                groups.append((group_value, positions))
        if not groups:
            raise ValueError(
                f"no groups contain more than max_lag={total_lag} observations"
            )

        row_positions = np.concatenate([pos[total_lag:] for _, pos in groups])
        inputs, raw = {}, {}

        for name, spec in self.cfg.terms.items():
            if isinstance(spec, SurfaceTermSpec):
                series = df[name].to_numpy(dtype=float)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(series)
                scaled_all = ensemble[0].term(name)._to_scaled(series)
                windows = []
                for _, pos in groups:
                    gvals = scaled_all[pos]
                    gw = make_windows(gvals, spec.lag_max)
                    # gw current rows begin at local spec.lag_max; align all
                    # surfaces to the common max lag used by the model sample.
                    gw = gw[total_lag - spec.lag_max:]
                    windows.append(gw)
                inputs[name] = torch.from_numpy(np.ascontiguousarray(np.concatenate(windows)))
                raw[name] = series[row_positions]

            elif isinstance(spec, SmoothTermSpec):
                series = df[name].to_numpy(dtype=float)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(series)
                vals = ensemble[0].term(name)._to_scaled(series)[row_positions]
                inputs[name] = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
                raw[name] = series[row_positions]

            elif isinstance(spec, TrendTermSpec):
                for m in ensemble:
                    m.term(name).fit_scaling()
                vals = self._trend_values(df, row_positions, time_col)
                inputs[name] = torch.from_numpy(vals).view(-1, 1)
                raw[name] = vals

            elif isinstance(spec, CategoricalTermSpec):
                source = self._source_col(name, spec)
                order = self._category_order(name, spec, df, ensemble, fit_metadata=fit_scaling)
                enc = {cat: i for i, cat in enumerate(order)}
                mapped_all = df[source].map(enc).to_numpy()
                mapped = mapped_all[row_positions]
                if pd.isna(mapped).any():
                    raise ValueError(f"unmapped category in '{source}'; check order")
                inputs[name] = torch.tensor(mapped.astype(np.int64), dtype=torch.long)
                raw[name] = df[source].to_numpy()[row_positions]

            else:
                raise NotImplementedError(f"no processor branch for {type(spec).__name__}")

        y = torch.tensor(df[target_col].to_numpy(dtype=float)[row_positions],
                         dtype=torch.float32).view(-1, 1)
        return self._pack(df, inputs, y, raw, row_positions, "grouped")

    # ------------------------------------------------------------------
    # PRELAGGED: read lag matrices created by the external research pipeline
    # ------------------------------------------------------------------
    def _prepare_prelagged(self, df, ensemble, target_col, time_col, fit_scaling):
        row_positions = np.arange(len(df), dtype=int)
        inputs, raw = {}, {}

        for name, spec in self.cfg.terms.items():
            if isinstance(spec, SurfaceTermSpec):
                cols = [f"{name}_lag{i}" for i in range(spec.lag_max + 1)]
                mat = df[cols].to_numpy(dtype=float)
                fit_values = mat.reshape(-1)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(fit_values)
                scaled = ensemble[0].term(name)._to_scaled(mat)
                inputs[name] = torch.tensor(scaled, dtype=torch.float32)
                raw[name] = mat[:, 0]

            elif isinstance(spec, SmoothTermSpec):
                series = df[name].to_numpy(dtype=float)
                if fit_scaling:
                    for m in ensemble:
                        m.term(name).fit_scaling(series)
                vals = ensemble[0].term(name)._to_scaled(series)
                inputs[name] = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
                raw[name] = series

            elif isinstance(spec, TrendTermSpec):
                for m in ensemble:
                    m.term(name).fit_scaling()
                vals = self._trend_values(df, row_positions, time_col)
                inputs[name] = torch.from_numpy(vals).view(-1, 1)
                raw[name] = vals

            elif isinstance(spec, CategoricalTermSpec):
                source = self._source_col(name, spec)
                order = self._category_order(name, spec, df, ensemble, fit_metadata=fit_scaling)
                enc = {cat: i for i, cat in enumerate(order)}
                mapped = df[source].map(enc).to_numpy()
                if pd.isna(mapped).any():
                    raise ValueError(f"unmapped category in '{source}'; check order")
                inputs[name] = torch.tensor(mapped.astype(np.int64), dtype=torch.long)
                raw[name] = df[source].to_numpy()

            else:
                raise NotImplementedError(f"no processor branch for {type(spec).__name__}")

        y = torch.tensor(df[target_col].to_numpy(dtype=float),
                         dtype=torch.float32).view(-1, 1)
        return self._pack(df, inputs, y, raw, row_positions, "prelagged")

    def _pack(self, df, inputs, y, raw, row_positions, input_type):
        n = int(len(row_positions))
        for name, x in inputs.items():
            if x.shape[0] != n:
                raise RuntimeError(
                    f"prepared term '{name}' has {x.shape[0]} rows; expected {n}"
                )
        original = df["__dlnam_original_index__"].to_numpy()[row_positions]
        return PreparedData(
            inputs=inputs,
            y=y,
            raw=raw,
            n_samples=n,
            row_index=original,
            input_type=input_type,
            category_orders=dict(self.category_orders),
        )

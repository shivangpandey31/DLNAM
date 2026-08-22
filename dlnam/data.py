"""
data.py — raw DataFrame -> model-ready inputs.

Turns a DataFrame into the {term_name: tensor} dict that Trainer.fit and
EffectExtractor consume, by dispatching on each term's spec:
  SurfaceTermSpec     -> scaled lag-window matrix (B, lag_max+1)
  SmoothTermSpec      -> scaled confounder column (B, 1)
  TrendTermSpec       -> normalised time in [0, 1]   (B, 1)
  CategoricalTermSpec -> integer level indices       (B,)  long

Scaling travels WITH the terms (via fit_scaling), not a global registry: the
processor fits each member's terms from the data and uses them to transform, so
EffectExtractor/visualisation later get correct raw<->scaled mapping for free.

Memory notes:
  * windows are built from VIEWS (sliding_window_view + reversed-column view)
    with a single float32 contiguous materialisation — no double copy, no
    float64 intermediate.
  * the dominant training-time memory cost is forward-pass activations, not this
    storage; control it with TrainConfig.batch_fraction (minibatch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import (ModelConfig, SurfaceTermSpec, SmoothTermSpec,
                     TrendTermSpec, CategoricalTermSpec)


def make_windows(series: np.ndarray, lag_max: int) -> np.ndarray:
    """(T,) series -> (T-lag_max, lag_max+1) where column j is lag j.

    Built from views; one contiguous float32 copy at the end. Sample i maps to
    time i+lag_max: col 0 = current value, col j = value j steps earlier.
    """
    s = np.ascontiguousarray(series, dtype=np.float32)
    asc = np.lib.stride_tricks.sliding_window_view(s, lag_max + 1)  # view
    win = asc[:, ::-1]                                              # view (flip)
    return np.ascontiguousarray(win)                               # one copy


@dataclass
class PreparedData:
    inputs: dict          # term name -> model-ready tensor (scaled)
    y: torch.Tensor       # (N, 1)
    raw: dict             # term name -> aligned raw np.ndarray (reference)
    n_samples: int


class DataProcessor:
    def __init__(self, model_config: ModelConfig):
        self.cfg = model_config

    def _required_columns(self, target_col: str) -> list[str]:
        cols = [target_col]
        for name, spec in self.cfg.terms.items():
            if isinstance(spec, TrendTermSpec):
                continue                                   # synthetic, no column
            cols.append(name)
        return cols

    def prepare(self, df: pd.DataFrame, ensemble, target_col: str) -> PreparedData:
        """df -> PreparedData, also fitting scaling on EVERY ensemble member's
        terms (identical, deterministic). `ensemble` is a list of DLNAM."""
        df = df.dropna(subset=self._required_columns(target_col)).reset_index(drop=True)

        surface_lags = [s.lag_max for s in self.cfg.terms.values()
                        if isinstance(s, SurfaceTermSpec)]
        total_lag = max(surface_lags) if surface_lags else 0
        n = len(df) - total_lag
        if n <= 0:
            raise ValueError("not enough rows after lag alignment")

        inputs, raw = {}, {}

        for name, spec in self.cfg.terms.items():
            if isinstance(spec, SurfaceTermSpec):
                series = df[name].to_numpy(dtype=float)
                for m in ensemble:
                    m.term(name).fit_scaling(series)
                scaled = ensemble[0].term(name)._to_scaled(series)
                win = make_windows(scaled, spec.lag_max)            # (T-Lk, Lk+1)
                win = win[total_lag - spec.lag_max:]                # align tails
                inputs[name] = torch.from_numpy(win)
                raw[name] = series[total_lag:]

            elif isinstance(spec, SmoothTermSpec):
                series = df[name].to_numpy(dtype=float)
                for m in ensemble:
                    m.term(name).fit_scaling(series)
                scaled = ensemble[0].term(name)._to_scaled(series)[total_lag:]
                inputs[name] = torch.tensor(scaled, dtype=torch.float32).view(-1, 1)
                raw[name] = series[total_lag:]

            elif isinstance(spec, TrendTermSpec):
                for m in ensemble:
                    m.term(name).fit_scaling()
                t = np.linspace(0.0, 1.0, n, dtype=np.float32)
                inputs[name] = torch.from_numpy(t).view(-1, 1)
                raw[name] = t

            elif isinstance(spec, CategoricalTermSpec):
                enc = {cat: i for i, cat in enumerate(spec.order)}
                idx = df[name].map(enc).to_numpy()[total_lag:]
                if np.isnan(idx.astype(float)).any():
                    raise ValueError(f"unmapped category in '{name}'; check order")
                inputs[name] = torch.tensor(idx.astype(np.int64), dtype=torch.long)
                raw[name] = idx

            else:
                raise NotImplementedError(f"no processor branch for {type(spec).__name__}")

        y = torch.tensor(df[target_col].to_numpy(dtype=float)[total_lag:],
                         dtype=torch.float32).view(-1, 1)
        return PreparedData(inputs=inputs, y=y, raw=raw, n_samples=n)

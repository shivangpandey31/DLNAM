"""Shared output helpers for paper experiment runners.

The runners should save enough structured data to recreate tables and figures
without scraping the terminal. These helpers keep the JSON schema, CSV summaries,
and environment metadata consistent across MC and real-data runs.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REGIONS = ("tot", "int", "bnd")
RESULTS_DIRNAME = "results"


def results_dir(base: str | Path) -> Path:
    """Directory for manuscript-facing result JSONs and figures."""
    path = Path(base) / RESULTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON conversion for numpy/torch/path objects used in result bundles."""
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass
    raise TypeError(type(obj).__name__)


def collect_environment() -> dict[str, Any]:
    """Small reproducibility block saved with every experiment result."""
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "torch", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _with_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    out.setdefault("environment", collect_environment())
    return out



def load_json_if_exists(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_with_metadata(payload), f, indent=2, default=json_default)
        f.write("\n")
    return path


def save_result_bundle(
    path: str | Path,
    *,
    kind: str,
    settings: Mapping[str, Any],
    models: Sequence[str],
    results: Mapping[str, Any],
    scenarios: Sequence[str] | None = None,
    exposures: Sequence[str] | None = None,
    regions: Sequence[str] = REGIONS,
    boundary: Mapping[str, Any] | None = None,
    curves: Mapping[str, Any] | None = None,
    **extras: Any,
) -> Path:
    """Save the canonical experiment bundle used by MC-style runners."""
    bundle: dict[str, Any] = {
        "kind": kind,
        "settings": dict(settings),
        "models": list(models),
        "regions": list(regions),
        "boundary": boundary or {},
        "results": results,
    }
    if scenarios is not None:
        bundle["scenarios"] = list(scenarios)
    if exposures is not None:
        bundle["exposures"] = list(exposures)
    if curves is not None:
        bundle["curves"] = curves
    for key, value in extras.items():
        if value is not None:
            bundle[key] = value
    return save_json(path, bundle)


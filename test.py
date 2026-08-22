"""Import-sanity check for the DLNAM Python codebase.

Run this from the "Python codes" directory to verify the core project
modules and the third-party Python packages used across scripts import cleanly.

Exit codes:
    0  all imports succeeded
    1  one or more imports failed
"""
from __future__ import annotations

import importlib
import sys
from typing import Iterable

CORE_IMPORTS = [
    "dlnam",
    "dlnam_sim",
    "dlnam_bench",
]

THIRD_PARTY_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "torch",
    "tqdm",
    "joblib",
    "pyarrow",
    "fastparquet",
    "openpyxl",
    "xlrd",
    "IPython",
    "jupyterlab",
    "notebook",
]

OPTIONAL_IMPORTS = [
    "h5py",
    "xarray",
    "netCDF4",
]


def _import_many(names: Iterable[str], *, optional: bool = False) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "version not exposed")
            print(f"{name:15s} OK  {version}")
        except Exception as exc:
            status = "OPTIONAL" if optional else "FAILED"
            print(f"{name:15s} {status}  {exc!r}")
            if not optional:
                failures.append((name, repr(exc)))
    return failures


def main() -> int:
    print("Python:", sys.version)
    print()

    failures = []
    print("== Core project packages ==")
    failures.extend(_import_many(CORE_IMPORTS))

    print()
    print("== Third-party packages ==")
    failures.extend(_import_many(THIRD_PARTY_IMPORTS))

    print()
    print("== Optional packages ==")
    _import_many(OPTIONAL_IMPORTS, optional=True)

    if failures:
        print()
        print("Import failures:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1

    print()
    print("All required imports passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

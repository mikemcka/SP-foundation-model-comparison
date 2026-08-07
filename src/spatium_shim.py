"""Make Spatium's `src/` importable without installing NVIDIA Merlin.

Spatium ships as a bare source tree: no `pyproject.toml`, no `setup.py`, no
install docs, and its modules import each other by bare name (`from model import
spaProFormer`), so `Spatium/src` has to be on `sys.path` as a *root* rather than
imported as a package.

The Merlin stub is the other half. `dataloader.py` imports `merlin.io`,
`merlin.dataloader.torch`, `merlin.dtypes` and `merlin.schema` at module scope,
but only its **parquet** path ever touches them; the few-shot path this
comparison uses (`FewShotTokenDataset` / `MerlinFewShotDataModule`) is pure zarr
+ torch. Merlin is a heavy, CUDA-pinned stack that does not co-resolve with
torch 2.6, so installing it to satisfy four unused imports would mean rebuilding
the environment around a dependency no code path here executes.

The stub therefore satisfies the imports and nothing more: every attribute it
hands out raises on *use*, so if a future change does route through Merlin it
fails loudly at the call site instead of silently doing the wrong thing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

SPATIUM_SRC = Path(__file__).resolve().parents[1] / "Spatium" / "src"


class _MerlinUnavailable:
    """Placeholder that raises if anything actually tries to use Merlin."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            f"{self._path} was called, but NVIDIA Merlin is deliberately not "
            "installed in this environment — only Spatium's zarr-backed "
            "few-shot path is used here. See src/spatium_shim.py."
        )

    def __getattr__(self, name: str) -> _MerlinUnavailable:
        return _MerlinUnavailable(f"{self._path}.{name}")


def _stub_getattr(name: str, _module: str) -> _MerlinUnavailable:
    """Serve stub attributes, but stay invisible to introspection.

    Dunders must raise ``AttributeError`` rather than return a placeholder:
    ``inspect`` walks every entry in ``sys.modules`` looking at ``__file__`` and
    friends (torch does this during import), and a stub that answers those
    lookups with a callable gets called by the interpreter's own machinery long
    before any Spatium code runs.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _MerlinUnavailable(f"{_module}.{name}")


def _install_merlin_stub() -> None:
    """Register stub `merlin.*` modules so `dataloader.py` imports cleanly."""
    if "merlin" in sys.modules:
        return

    for name in (
        "merlin",
        "merlin.io",
        "merlin.dataloader",
        "merlin.dataloader.torch",
        "merlin.dtypes",
        "merlin.schema",
    ):
        module = types.ModuleType(name)
        # Point introspection at this file: it is the honest answer for where
        # these modules come from, and it keeps `inspect.getsourcefile` happy.
        module.__file__ = __file__
        module.__getattr__ = lambda attr, _n=name: _stub_getattr(attr, _n)  # type: ignore[attr-defined]
        sys.modules[name] = module
    sys.modules["merlin"].__path__ = []  # type: ignore[attr-defined]
    sys.modules["merlin.dataloader"].__path__ = []  # type: ignore[attr-defined]

    # `dataloader.py` does `from merlin.dtypes import boolean, float32, ...` and
    # `from merlin.schema import ColumnSchema, Schema`, which need real
    # attributes to bind at import time rather than going through __getattr__.
    for attr in ("boolean", "float32", "int64", "int32", "string"):
        setattr(sys.modules["merlin.dtypes"], attr, _MerlinUnavailable(f"merlin.dtypes.{attr}"))
    for attr in ("ColumnSchema", "Schema"):
        setattr(sys.modules["merlin.schema"], attr, _MerlinUnavailable(f"merlin.schema.{attr}"))
    sys.modules["merlin.dataloader.torch"].Loader = _MerlinUnavailable(
        "merlin.dataloader.torch.Loader"
    )
    sys.modules["merlin.io"].Dataset = _MerlinUnavailable("merlin.io.Dataset")


def setup() -> Path:
    """Put Spatium's source tree on the path and stub Merlin.

    Returns:
        The ``Spatium/src`` directory that was added to ``sys.path``.

    Raises:
        FileNotFoundError: If the Spatium checkout is missing.
    """
    if not SPATIUM_SRC.exists():
        raise FileNotFoundError(
            f"No Spatium source tree at {SPATIUM_SRC}. Clone "
            "https://github.com/ploughhh/Spatium into the repo root."
        )
    _install_merlin_stub()
    if str(SPATIUM_SRC) not in sys.path:
        sys.path.insert(0, str(SPATIUM_SRC))
    return SPATIUM_SRC

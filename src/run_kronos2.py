"""KRONOS2: per-cell CLS embeddings from cell-centred patches.

Runs in the `sp-coral` env, on a GPU.

KRONOS2 is a marker-aware ViT: it takes the 64 px box around a cell together
with the *names* of the channels in that box, and returns one 768-d CLS vector.
The marker-aware z-score normalisation lives inside the model (`model.preprocess`),
so CORAL hands it float32 `[0, 1]` patches and does not normalise itself — this
script must not add its own normalisation on top or the embeddings stop matching
the published gold standard.

Reproducibility notes carried over from CORAL: the model runs in fp32, and the
gold-standard features were produced at `--batch-size 16`. On GPU the batch size
can shift embeddings by ~1e-4 through cuBLAS kernel selection, so the batch size
is pinned here rather than tuned for throughput.

Weights come from the gated `MahmoodLab/KRONOS2` Hub repo; `HF_TOKEN` and
`HF_HOME` must be set (see `env/secrets.env`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import protocol  # noqa: E402

JOB_DIR = protocol.DATA_DIR / "processed"
PANEL_YAML = protocol.DATA_DIR / "panel.yaml"
CELL_SLUG = f"cell_{protocol.MPP}mpp_{protocol.CELL_PATCH_SIZE}px"

# The batch size the published KRONOS2 features were produced at. Not a
# throughput knob — see the module docstring.
GOLD_BATCH_SIZE = 16


def check_credentials() -> None:
    """Fail early and legibly if the gated-Hub credentials are missing."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. MahmoodLab/KRONOS2 is a manual-approval "
            "gated repo, so the download 401s without it. "
            "Run: source env/secrets.env"
        )


def loader_workers(reserve: int = 2, cap: int = 12) -> int:
    """Pick a patch-loader worker count from the cores this job actually has.

    Reads Slurm's allocation rather than ``os.cpu_count()``, which reports the
    whole node and would oversubscribe a shared one. Feeding the GPU is the
    point: 152k random 64px reads out of a zarr on a network filesystem will
    otherwise leave the forward pass idling.
    """
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    return max(2, min(cap, allocated - reserve))


def extract(device: str) -> None:
    """Encode every cell patch with KRONOS2 via the CORAL CLI."""
    cmd = [
        "coral", "extract",
        "--job-dir", str(JOB_DIR),
        "--extractor", "KRONOS2",
        "--patches", CELL_SLUG,
        "--subset", str(PANEL_YAML),
        "--batch-size", str(GOLD_BATCH_SIZE),
        "--num-workers", str(loader_workers()),
        "--device", device,
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def export() -> None:
    """Save the stored embeddings in the shared ``.npz`` feature format."""
    from coral import CoralSlide
    from coral.config import PatchConfig

    slide = CoralSlide.open(JOB_DIR / "raw_image.zarr")
    cfg = PatchConfig(patch_size=protocol.CELL_PATCH_SIZE, mode="cell_centered")
    ds = slide.features("KRONOS2", cfg, suffix="markers_panel")

    out = protocol.RESULTS / "features" / "kronos2.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        cell_id=ds.coords["cell_id"].values,
        features=np.asarray(ds["features"].values, dtype=np.float32),
    )
    print(f"wrote KRONOS2 features {ds['features'].shape} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip extraction; only re-export already-stored features.",
    )
    args = parser.parse_args()

    if not args.export_only:
        check_credentials()
        extract(args.device)
    export()


if __name__ == "__main__":
    main()

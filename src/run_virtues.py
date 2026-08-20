"""VirTues: per-cell tokens from a whole-tissue encoding.

Three stages, three different machines, handed off through files:

    --stage panel     sp-coral,   CPU        export the panel as an OME-TIFF
    --stage markers   sp-virtues, CPU + net  UniProt -> ESM-2 marker embeddings
    --stage embed     sp-virtues, Ampere GPU cell tokens -> features/virtues.npz

VirTues reads a *tissue*, not a cell. It tiles the whole image, encodes each tile
with alternating spatial and marker attention, and emits one summary token per
8x8 patch; a cell token is the pixel-overlap-weighted mean of the patch tokens
that cell intersects (paper Methods, "Aggregation into cell-, niche- and
tissue-level representations"). Two consequences shape everything below.

**Its receptive field is much larger than the other encoders'.** KRONOS2 and
DeepCell Types read a 64 px box at this slide's native 0.37 mpp — about 24 um of
tissue. VirTues resamples to 1.0 mpp and runs spatial attention over 128 px
crops, so every patch token has already attended over ~128 um. Its probe row is
therefore NOT pixel-matched to the others: it sees neighbourhood context they
cannot. That is a property of the model, not a protocol slip, and it is reported
as a caveat rather than corrected away — there is no way to shrink its receptive
field without retraining it. See RESULTS.md.

**It has no marker vocabulary at all.** A channel is identified only by an ESM-2
embedding of its target protein's amino acid sequence — no name matching, no
alias table inside the model, no fallback. So DAPI is unusable: DNA has no amino
acid sequence and there is nothing honest to embed. VirTues sees 17 of the 18
panel markers, exactly as Spatium sees 16 for the same class of reason. The
nuclear-proxy stand-in (DAPI presented as Histone H3) is deliberately NOT used;
see src/common/virtues_marker_uniprot.csv.

The encoder is not reimplemented here. VirTues-Nextflow already carries a
heavily-annotated, memory-bounded implementation of exactly this readout in
`bin/virtues_embeddings.py`, validated against the authors' own demo notebook, so
this script stages that script's inputs and converts its output rather than
writing a second copy of the same math for the two to drift apart. The one thing
worth knowing about its internals: upstream's `compute_cell_tokens` holds every
crop's tokens plus a tensor per (cell, patch) pair in memory, which is tens of GB
on a slide this size; the wrapper does the identical arithmetic as a running
`index_add_` and stays around 285 MB.

Weights are public — `bunnelab/virtues` needs no token, unlike KRONOS2 and
DeepCell Types. `virtues-sp32` is the pipeline's default and is CC BY-NC 4.0
(academic use); `virtues-sp31` is the same model minus one dataset under MIT, for
commercial work. This benchmark is academic, so sp32 stands.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import protocol  # noqa: E402

# The wrapper scripts and the shipped marker table live in the Nextflow pipeline;
# the model source is a clone of the authors' repo. Both are inputs to this
# stage, so both are overridable rather than hardcoded — but they default to
# where they actually are, so the common case needs no environment at all.
NEXTFLOW_REPO = Path(
    os.environ.get(
        "VIRTUES_NEXTFLOW", "/vast/scratch/users/mckay.m/VirTues-Nextflow"
    )
)
VIRTUES_REPO = Path(
    os.environ.get("VIRTUES_REPO", protocol.REPO / "virtues")
)

JOB_DIR = protocol.DATA_DIR / "processed"
MASK_TIFF = protocol.CHL_DIR / "segmentation" / "raw_image.tiff"

WORK = protocol.RESULTS / "virtues"
PANEL_TIFF = WORK / "panel.ome.tiff"
CHANNELS_CSV = WORK / "channels.csv"
MARKER_EMBEDDINGS = WORK / "marker_embeddings"
MARKER_REPORT = WORK / "marker_resolution.txt"
WEIGHTS_DIR = WORK / "weights"

MARKER_OVERRIDE = Path(__file__).resolve().parent / "common" / "virtues_marker_uniprot.csv"

HF_REPO = "bunnelab/virtues"
MODEL_NAME = "virtues-sp32"

# VirTues' own native resolution. It takes no mpp argument and does no internal
# resampling — its training corpus is stored at 1.0 um/px, so its 8 px patch IS
# an 8 um patch and the wrapper resamples the stack to match. We hand it this
# slide's true 0.37 mpp and let it do that, which is the same policy every other
# model in this benchmark gets (see CLAUDE.md, "No resample to 0.5 um/px").
PREFIX = "chl_codex"


def _wrapper(name: str) -> Path:
    """Locate one of VirTues-Nextflow's bin/ scripts, or say what is missing."""
    path = NEXTFLOW_REPO / "bin" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This stage drives VirTues-Nextflow's wrapper "
            "scripts; point VIRTUES_NEXTFLOW at that checkout."
        )
    return path


def _marker_table() -> Path:
    """The shipped marker -> UniProt table, which the override layers onto."""
    path = NEXTFLOW_REPO / "assets" / "marker_uniprot.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (VIRTUES_NEXTFLOW wrong?)")
    return path


def _check_virtues_repo() -> Path:
    """Fail early if the model source is not staged.

    VirTues is imported off a clone rather than pip-installed: its pyproject
    declares only the top-level package and it ships no ``__init__.py`` for
    ``virtues.utils``, so a non-editable install does not carry what the wrapper
    imports. The authors' own instruction is ``pip install -e .``; staging the
    tree does the same thing and pins the exact source that ran.
    """
    if not (VIRTUES_REPO / "virtues").is_dir():
        raise FileNotFoundError(
            f"No VirTues source at {VIRTUES_REPO}. Run:\n"
            f"  git clone https://github.com/bunnelab/virtues.git {VIRTUES_REPO}"
        )
    return VIRTUES_REPO


def run(cmd: list[str], *, env: dict | None = None) -> None:
    """Run a wrapper script, echoing the command so logs are reproducible."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


# --- stage: panel ----------------------------------------------------------


def export_panel() -> None:
    """Write the 18-marker panel as a named-channel OME-TIFF.

    Channels come from the ingested CORAL slide, not the raw TIFFs, so VirTues
    reads exactly the planes KRONOS2 and DeepCell Types read, in the same order.

    All 18 are written, DAPI included, even though VirTues cannot use it. Letting
    its own ``resolve_markers`` drop the channel means the decision is recorded
    in the marker report by the model's own machinery, rather than being
    pre-applied here where a later reader would have no way to see it happened.

    Pixels stay uint16 and unscaled. VirTues standardises each channel itself —
    clip at the 99th percentile, ``log1p``, 3x3 Gaussian blur, z-score, all
    three statistics restricted to tissue-masked pixels per the paper's
    Methods ("Dataset preprocessing") — and the z-score after the log removes
    any global scale factor, so normalising first would only interpolate the
    intensities twice.
    """
    import tifffile
    from coral import CoralSlide

    slide = CoralSlide.open(JOB_DIR / "raw_image.zarr")
    available = list(slide.markers)
    missing = [m for m in protocol.PANEL if m not in available]
    if missing:
        raise ValueError(f"panel markers absent from the slide: {missing}")

    idx = [available.index(m) for m in protocol.PANEL]
    WORK.mkdir(parents=True, exist_ok=True)

    stack = np.asarray(slide.image[idx].values, dtype=np.uint16)
    print(f"panel stack {stack.shape} {stack.dtype}")
    tifffile.imwrite(
        PANEL_TIFF,
        stack,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(protocol.PANEL)}},
        ome=True,
        bigtiff=True,
    )
    print(f"wrote {PANEL_TIFF} ({PANEL_TIFF.stat().st_size / 1e9:.2f} GB)")

    with open(CHANNELS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "index", "channel"])
        for i, name in enumerate(protocol.PANEL):
            writer.writerow([PREFIX, i, name])
    print(f"wrote {CHANNELS_CSV}")

    # The mask must be pixel-for-pixel the image's size or every cell misaligns.
    # The wrapper checks this too, but failing here costs seconds instead of a
    # queued GPU job.
    mask_shape = tifffile.TiffFile(MASK_TIFF).series[0].shape
    if tuple(mask_shape[-2:]) != stack.shape[-2:]:
        raise ValueError(
            f"mask is {mask_shape[-2:]} but the panel is {stack.shape[-2:]}"
        )
    print(f"mask {MASK_TIFF.name} {mask_shape} matches the panel")


# --- stage: markers --------------------------------------------------------


def build_marker_embeddings() -> None:
    """Resolve the panel to UniProt and embed each accession with ESM-2.

    Needs outbound network twice — UniProt for the sequences, then the ESM-2
    weights — which is why it is its own stage: cluster GPU nodes frequently have
    none.

    Built ONCE, into one directory, and then left alone. ``load_marker_embeddings``
    stacks the directory's ``.pt`` files in *sorted filename* order and
    ``load_marker_embedding_dict`` derives the accession -> row index map from
    that same order, so which row a protein occupies depends on which other files
    are present. Rebuilding this directory with a different panel would silently
    reindex every channel; nothing would error and the embeddings would just be
    wrong.

    ``--allow-unmapped-markers`` is deliberately NOT passed: a channel with no
    table entry should stop the run, because VirTues has no way to be told what
    it is and would quietly embed a smaller panel than the image carries.
    """
    _check_virtues_repo()
    if not CHANNELS_CSV.exists():
        raise FileNotFoundError(
            f"{CHANNELS_CSV} not found; run --stage panel first (env sp-coral)."
        )
    run([
        sys.executable, _wrapper("virtues_markers.py"),
        "--channels", CHANNELS_CSV,
        "--virtues-repo", _check_virtues_repo(),
        "--marker-table", _marker_table(),
        "--marker-uniprot", MARKER_OVERRIDE,
        "--outdir", MARKER_EMBEDDINGS,
        "--report", MARKER_REPORT,
        "--device", "cpu",
    ], env={**os.environ, "PYTHONPATH": str(NEXTFLOW_REPO / "bin")})

    accessions = sorted(p.stem for p in MARKER_EMBEDDINGS.glob("*.pt"))
    print(f"\n{len(accessions)} marker embeddings: {accessions}")
    if len(accessions) != len(protocol.PANEL) - 1:
        print(
            f"NOTE: {len(protocol.PANEL)} panel markers -> {len(accessions)} "
            "embeddings. One drop (DAPI) is expected and intended; anything "
            "else means the marker table changed.",
            file=sys.stderr,
        )


def fetch_weights() -> Path:
    """Download the released checkpoint once and return its path."""
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"{MODEL_NAME}/model.safetensors",
        local_dir=str(WEIGHTS_DIR),
    )
    print(f"staged {path}")
    return Path(path)


# --- stage: embed ----------------------------------------------------------


def encode_cells(device: str, chunk_size: int) -> None:
    """Run the cell-level pass and leave a parquet of per-cell tokens."""
    weights = WEIGHTS_DIR / MODEL_NAME / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} not found; run --stage markers first (it also fetches "
            "the checkpoint, on a node with network)."
        )
    if not MARKER_EMBEDDINGS.is_dir():
        raise FileNotFoundError(
            f"{MARKER_EMBEDDINGS} not found; run --stage markers first."
        )

    WORK.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, _wrapper("virtues_embeddings.py"),
        "--tiff", PANEL_TIFF,
        # The published mask, not a re-segmentation: the ground-truth labels are
        # defined on these exact ids, and VirTues pools by mask pixels, so the
        # ids come straight through to the token table.
        "--cells", MASK_TIFF,
        "--marker-embeddings", MARKER_EMBEDDINGS,
        "--weights", weights,
        "--config", _check_virtues_repo() / "configs" / "base_config.yaml",
        "--virtues-repo", _check_virtues_repo(),
        "--marker-table", _marker_table(),
        "--marker-uniprot", MARKER_OVERRIDE,
        "--mpp", protocol.MPP,
        "--prefix", WORK / PREFIX,
        # Only the cell level is scored. Niche and tissue tokens are real
        # readouts but there is nothing to compare them against here: every other
        # encoder in this benchmark emits one vector per cell.
        "--levels", "cell",
        "--chunk-size", chunk_size,
        "--device", device,
    ], env={**os.environ, "PYTHONPATH": str(NEXTFLOW_REPO / "bin")})


def export_features() -> None:
    """Convert the token parquet into the shared ``.npz`` feature format.

    The parquet carries every cell in the mask, which is a superset of the
    canonical table — cells that are unlabelled, artifacts, or too close to the
    edge to have had a 64 px patch. ``protocol.align_features`` reindexes by id,
    so the superset is fine and the subset would not be; this only checks that
    the canonical cells are all actually in there.
    """
    import pandas as pd

    parquet = WORK / f"{PREFIX}_cell_tokens.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found; --stage embed first.")

    frame = pd.read_parquet(parquet)
    dims = [c for c in frame.columns if c.startswith("dim_")]
    features = frame[dims].to_numpy(dtype=np.float32)
    cell_id = frame["cell_id"].to_numpy()
    print(f"cell tokens {features.shape} for {len(cell_id):,} mask cells")

    cells = protocol.load_cells()
    absent = np.setdiff1d(cells.index.to_numpy(), cell_id)
    if len(absent):
        raise ValueError(
            f"{len(absent):,} of {len(cells):,} canonical cells have no VirTues "
            "token; the mask ids may have drifted."
        )

    out = protocol.RESULTS / "features" / "virtues.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cell_id=cell_id, features=features)
    print(f"wrote VirTues features {features.shape} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("panel", "markers", "embed"), required=True,
        help="panel: sp-coral. markers: sp-virtues + network. embed: Ampere GPU.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--chunk-size", type=int, default=32, help="Crops per forward pass."
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="With --stage embed: skip encoding, only re-export the parquet.",
    )
    args = parser.parse_args()

    if args.stage == "panel":
        export_panel()
    elif args.stage == "markers":
        build_marker_embeddings()
        fetch_weights()
    else:
        if not args.export_only:
            encode_cells(args.device, args.chunk_size)
        export_features()


if __name__ == "__main__":
    main()

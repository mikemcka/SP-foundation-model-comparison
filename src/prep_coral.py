"""Shared preprocessing: raw Zenodo download -> canonical cell table.

Runs in the `sp-coral` env. Everything here is model-agnostic on purpose — it
produces the one slide, the one cell mask, the one label set and the one fold
assignment that all three foundation models are then measured against. If a
model needs its input shaped differently (Spatium wants an expression matrix,
DeepCell Types wants the raw stack), it derives that from these outputs rather
than re-reading the raw data its own way.

Steps, in CORAL's documented order:

1. **Arrange** the Zenodo extract into the cohort layout the CLI expects.
2. **Ingest** the per-channel TIFFs into a canonical CORAL slide at native
   0.37 mpp, canonicalising marker names as it goes.
3. **Import** the published segmentation mask and MAPS cell-type labels.
   Cellpose is deliberately *not* run: every model must see the same cells, and
   the published mask is what the ground-truth labels were assigned on.
4. **Patch** one 64 px box per cell centroid.
5. **Extract** the mean-marker baseline over the 18-marker panel.
6. **Freeze** the canonical cell table (labels, centroids, quadrant folds).

Idempotent: each CORAL step records its status in the store and is skipped when
already complete, so re-running after a failure resumes rather than restarts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import protocol  # noqa: E402

ARCHIVE = protocol.DATA_DIR / "cHL_CODEX.zip"
COHORT = protocol.CHL_DIR / "cohort"
RAW_IMAGE = COHORT / "raw_image"
MASK_DIR = protocol.CHL_DIR / "segmentation"
LABELS_DIR = protocol.CHL_DIR / "labels"
ANNOT_CSV = protocol.CHL_DIR / "annotation_csv" / "cHL_CODEX_annotation.csv"
JOB_DIR = protocol.DATA_DIR / "processed"
PANEL_YAML = protocol.DATA_DIR / "panel.yaml"

CELL_SLUG = f"cell_{protocol.MPP}mpp_{protocol.CELL_PATCH_SIZE}px"
LABEL_SET = "maps"


# The four channel names CORAL's registry cannot auto-resolve on this slide.
# CORAL deliberately stops and asks rather than guessing, because the mapping is
# a scientific judgement, not a string-matching problem:
#
#   DAPI-01     -> DAPI         a per-cycle nuclear re-stain
#   Cytokeritin -> CYTOKERATIN  vendor typo
#   VISA        -> VISTA        a.k.a. VSIR / B7-H5
#   Collagen 4  -> COLLAGENiv   collagen IV specifically, NOT the generic
#                               COLLAGEN that CORAL suggests first
#
# Only DAPI-01 and Cytokeritin are in the 18-marker panel; the other two are
# fixed because ingest refuses to proceed with any channel unresolved.
MARKER_MAP_FIXES = {
    "DAPI-01": "DAPI",
    "Cytokeritin": "CYTOKERATIN",
    "VISA": "VISTA",
    "Collagen 4": "COLLAGENiv",
}


def loader_workers(reserve: int = 2, cap: int = 12) -> int:
    """Pick a patch-loader worker count from the cores this job actually has.

    Reads Slurm's allocation rather than ``os.cpu_count()``, which reports the
    whole node and would oversubscribe a shared one.

    Args:
        reserve: Cores to leave for the main process.
        cap: Upper bound — past this, workers contend on the filesystem
            rather than adding throughput.

    Returns:
        A worker count of at least 2.
    """
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    return max(2, min(cap, allocated - reserve))


def run(cmd: list[str], *, check: bool = True) -> int:
    """Run a CORAL CLI command, echoing it first.

    Args:
        cmd: The command and its arguments.
        check: Raise on a non-zero exit. Set ``False`` for the ingest review
            gate, whose non-zero exit is the documented "now go fix the marker
            map" signal rather than a failure.

    Returns:
        The process exit code.
    """
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def fix_marker_map() -> None:
    """Resolve the four flagged channels so ingest's second pass can proceed.

    ``coral ingest`` is a two-pass command: the first pass reads channel names
    only, writes ``marker_map.csv``, and stops at a review gate for any name it
    cannot resolve against its registry. This applies the decisions in
    :data:`MARKER_MAP_FIXES` and marks the rows resolved; the second ingest pass
    then reads pixels.
    """
    from coral.markers.marker_map import read_marker_map, write_marker_map

    marker_map = read_marker_map(JOB_DIR)
    for raw_name, canonical in MARKER_MAP_FIXES.items():
        row = marker_map["original_name"] == raw_name
        if not row.any():
            raise ValueError(
                f"{raw_name!r} is not in {JOB_DIR / 'marker_map.csv'}; the "
                "dataset's channel names have changed."
            )
        marker_map.loc[row, "mapped_canonical_name"] = canonical
        marker_map.loc[row, "status"] = "RESOLVED"

    unresolved = marker_map.loc[marker_map["status"] != "RESOLVED", "original_name"]
    if len(unresolved):
        raise ValueError(
            f"still unresolved after fixes: {list(unresolved)}. Add them to "
            "MARKER_MAP_FIXES with a deliberate canonical name."
        )
    write_marker_map(marker_map, JOB_DIR)
    print(f"\nresolved {len(MARKER_MAP_FIXES)} flagged markers: "
          + ", ".join(f"{k} -> {v}" for k, v in MARKER_MAP_FIXES.items()))


def arrange_dataset() -> None:
    """Unpack the archive and move it into the layout the CLI expects.

    The Zenodo archive ships ``raw_image/`` at the dataset root, but ``coral
    ingest`` takes a *cohort* directory (one entry per slide), and the cell
    pipeline pairs a mask to its slide **by filename** — ``raw_image/`` ingests
    to ``raw_image.zarr``, so the mask must be ``raw_image.tiff``. Both moves
    are one-time and guarded, so re-running is free.
    """
    if not protocol.CHL_DIR.exists():
        if not ARCHIVE.exists():
            raise FileNotFoundError(
                f"Neither {protocol.CHL_DIR} nor {ARCHIVE} exists. Download the "
                "dataset first (see CLAUDE.md)."
            )
        print(f"Extracting {ARCHIVE.name} ...", flush=True)
        with zipfile.ZipFile(ARCHIVE) as zf:
            zf.extractall(protocol.DATA_DIR)

    legacy_images = protocol.CHL_DIR / "raw_image"
    if legacy_images.exists() and not RAW_IMAGE.exists():
        COHORT.mkdir(exist_ok=True)
        shutil.move(str(legacy_images), str(RAW_IMAGE))

    legacy_mask = MASK_DIR / "cHL_CODEX_segmentation.tiff"
    slide_mask = MASK_DIR / "raw_image.tiff"
    if legacy_mask.exists() and not slide_mask.exists():
        shutil.move(str(legacy_mask), str(slide_mask))

    markers = sorted(p.stem for p in RAW_IMAGE.glob("*.tif*"))
    print(f"cohort : {COHORT}")
    print(f"markers: {len(markers)} channels")
    print(f"mask   : {slide_mask} ({'ok' if slide_mask.exists() else 'MISSING'})")


def stage_labels() -> pd.DataFrame:
    """Rewrite the MAPS annotation into CORAL's ``cell_id,label`` schema.

    ``coral cell --custom-label-path`` wants a directory of per-store CSVs named
    after each store, so this writes ``labels/raw_image.csv`` to pair with
    ``raw_image.zarr``.

    Returns:
        The staged label frame.
    """
    annot = pd.read_csv(ANNOT_CSV)
    labels = annot.rename(columns={"cellLabel": "cell_id", "cellType": "label"})[
        ["cell_id", "label"]
    ]
    LABELS_DIR.mkdir(exist_ok=True)
    labels.to_csv(LABELS_DIR / "raw_image.csv", index=False)
    print(f"\nstaged {len(labels):,} labels -> {LABELS_DIR / 'raw_image.csv'}")
    print(labels["label"].value_counts().to_string())
    return labels


def write_panel() -> None:
    """Write the 18-marker panel that every encoder extracts on."""
    PANEL_YAML.parent.mkdir(parents=True, exist_ok=True)
    PANEL_YAML.write_text(
        yaml.safe_dump(
            {"name": "panel", "channels": {"include": protocol.PANEL}},
            sort_keys=False,
        )
    )
    print(f"\nwrote {PANEL_YAML} ({len(protocol.PANEL)} markers)")


def coral_pipeline() -> None:
    """Ingest, import cells, patch, and extract the mean-marker baseline."""
    ingest_cmd = [
        "coral", "ingest",
        "--image-dir", COHORT,
        "--job-dir", JOB_DIR,
        "--mpp", str(protocol.MPP),
        "--nuclear-marker", protocol.NUCLEAR_MARKER,
    ]
    # Pass 1 builds marker_map.csv from channel names alone and stops at the
    # review gate (non-zero exit is expected here, not a failure). Pass 2 reads
    # the pixels once every name resolves. Re-running an already-ingested store
    # never re-reads pixels, so this is cheap on a resume.
    run(ingest_cmd, check=False)
    fix_marker_map()
    run(ingest_cmd)
    # Tissue segmentation is a formality here, not a filter. Cell-centred patch
    # sets are explicitly NOT tissue-restricted (`CoralSlide.extract_patches`
    # resolves the method for provenance only), but the `coral patch` CLI
    # resolves it eagerly and errors when no mask exists — so run the built-in
    # Otsu segmenter to satisfy it. Its quality cannot change which cells are
    # patched; the imported mask alone decides that.
    run([
        "coral", "tissue",
        "--job-dir", JOB_DIR,
        "--segmentation-method", "otsu",
    ])
    # Import the published mask + labels rather than segmenting: the ground
    # truth is defined on these exact cell ids, and re-segmenting would break
    # the join and change what every model is scored on.
    run([
        "coral", "cell",
        "--job-dir", JOB_DIR,
        "--custom-mask-path", MASK_DIR,
        "--custom-label-path", LABELS_DIR,
        "--label-set", LABEL_SET,
    ])
    run([
        "coral", "patch",
        "--job-dir", JOB_DIR,
        "--patch-size", str(protocol.CELL_PATCH_SIZE),
        "--mode", "cell",
    ])
    # Extraction is I/O-bound, not compute-bound: it is 152k random 64px reads
    # out of a zarr on a network filesystem, and the mean-marker "model" is an
    # average. CORAL's default of 4 loader workers is tuned for a GPU forward
    # pass it needs to keep fed; here the reads *are* the work, so scale the
    # workers with the cores the job actually has.
    run([
        "coral", "extract",
        "--job-dir", JOB_DIR,
        "--extractor", "mean_marker",
        "--patches", CELL_SLUG,
        "--subset", PANEL_YAML,
        "--num-workers", str(loader_workers()),
    ])


def build_cell_table() -> pd.DataFrame:
    """Freeze the canonical cell list: curated labels, centroids, folds.

    This is the artifact that makes the comparison fair — written once, read by
    every model script and by the evaluator. Curation and fold assignment
    follow :mod:`common.protocol`.

    The cell list is seeded from the **patch set**, not from the mask: a cell
    whose 64 px box would fall outside the image gets no patch, so it has no
    features and cannot be scored. Seeding from the mask instead would put cells
    in the canonical table that no encoder can supply a row for.

    Returns:
        The canonical table, also written to ``results/cells.parquet``.
    """
    from coral import CoralSlide
    from coral.config import PatchConfig

    slide = CoralSlide.open(JOB_DIR / "raw_image.zarr")
    _, height, width = slide.image.shape

    cfg = PatchConfig(patch_size=protocol.CELL_PATCH_SIZE, mode="cell_centered")
    patched_ids = slide.features("mean_marker", cfg, suffix="markers_panel").coords[
        "cell_id"
    ].values

    labels_df = pd.read_csv(slide.path / "cells" / "cell_labels.csv")
    labels_df = labels_df[labels_df["label_set"] == LABEL_SET].set_index("cell_id")
    centroids = pd.read_csv(slide.path / "cells" / "cell_centroids.csv").set_index(
        "cell_id"
    )

    # Centroids come from cell_centroids.csv, not from the patch coords: a
    # cell-centred patch stores its top-left corner, while the fold split needs
    # the cell's actual centre.
    cells = pd.DataFrame(
        {
            "label": labels_df["label"].reindex(patched_ids).to_numpy(),
            "x": centroids["x"].reindex(patched_ids).to_numpy(),
            "y": centroids["y"].reindex(patched_ids).to_numpy(),
        },
        index=pd.Index(patched_ids, name="cell_id"),
    )
    n_total = len(cells)
    print(f"\n{n_total:,} cells have a patch (of {len(centroids):,} in the mask)")

    # Drop unlabelled cells and segmentation artifacts, then merge the rare
    # CD8 subtype — merge last so CD8 absorbs it before codes are assigned.
    cells = cells[cells["label"].notna()]
    cells = cells[~cells["label"].isin(protocol.DROP_LABELS)]
    n_merged = int(cells["label"].isin(protocol.MERGE_LABELS).sum())
    cells["label"] = protocol.curate_labels(cells["label"])
    print(f"\n{len(cells):,} labelled non-artifact cells of {n_total:,} in the mask")
    print(f"merged {n_merged:,} '{list(protocol.MERGE_LABELS)[0]}' cells")

    found = sorted(cells["label"].unique())
    if found != protocol.CLASS_NAMES:
        raise ValueError(
            "class list drifted from the published benchmark:\n"
            f"  here:      {found}\n"
            f"  benchmark: {protocol.CLASS_NAMES}"
        )

    cells["code"] = pd.Categorical(
        cells["label"], categories=protocol.CLASS_NAMES
    ).codes.astype(np.int64)
    cells["quadrant"] = protocol.assign_quadrants(
        cells["x"].to_numpy(), cells["y"].to_numpy(), width, height
    )

    n_band = int((cells["quadrant"] == 0).sum())
    cells = cells[cells["quadrant"] > 0]
    print(f"dropped {n_band:,} cells in the {protocol.CELL_PATCH_SIZE}px guard band")
    print(
        cells["quadrant"].value_counts().sort_index().rename("cells per quadrant")
        .to_string()
    )

    protocol.RESULTS.mkdir(parents=True, exist_ok=True)
    cells.index.name = "cell_id"
    cells.to_parquet(protocol.CELLS_PARQUET)
    print(f"\nwrote {len(cells):,} canonical cells -> {protocol.CELLS_PARQUET}")
    print(cells["label"].value_counts().to_string())
    return cells


def export_mean_marker() -> None:
    """Save the mean-marker baseline in the shared ``.npz`` feature format."""
    from coral import CoralSlide
    from coral.config import PatchConfig

    slide = CoralSlide.open(JOB_DIR / "raw_image.zarr")
    cfg = PatchConfig(patch_size=protocol.CELL_PATCH_SIZE, mode="cell_centered")
    ds = slide.features("mean_marker", cfg, suffix="markers_panel")

    out = protocol.RESULTS / "features" / "mean_marker.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        cell_id=ds.coords["cell_id"].values,
        features=np.asarray(ds["features"].values, dtype=np.float32),
    )
    print(f"wrote mean-marker features {ds['features'].shape} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-coral",
        action="store_true",
        help="Skip the CORAL CLI steps and only rebuild the cell table.",
    )
    args = parser.parse_args()

    arrange_dataset()
    if not args.skip_coral:
        stage_labels()
        write_panel()
        coral_pipeline()
    build_cell_table()
    export_mean_marker()
    print("\nprep complete.")


if __name__ == "__main__":
    main()

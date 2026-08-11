"""The shared contract every model in this comparison is held to.

The whole point of the repo is that three foundation models are scored on the
*same cells, same labels, same spatial folds, same probe*. Anything a model is
free to vary — how it turns pixels into a vector — is the thing being measured;
everything else is pinned here, in one file, so a difference in score cannot be
a difference in protocol.

The protocol is CORAL's published cell-phenotyping benchmark (tutorial 5),
adopted wholesale rather than reinvented: 16 classes, quadrant folds with a
64 px guard band, a class-capped and class-weighted logistic probe with `C`
tuned by Optuna on validation only. `probe.py` imports CORAL's own helpers so
the numbers here land on the same footing as the published ones.

Row order is the load-bearing detail. `cells.parquet` (written once by
`prep_coral.py`) is the canonical cell list; every model writes a feature matrix
or a prediction keyed by `cell_id`, and `evaluate.py` reindexes onto this table.
No script may assume its own natural ordering matches another's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "CORAL" / "tutorials" / "example-data"
CHL_DIR = DATA_DIR / "cHL_CODEX"
STORE = DATA_DIR / "processed" / "raw_image.zarr"
MASK_TIFF = CHL_DIR / "segmentation" / "raw_image.tiff"
RESULTS = REPO / "results"

# --- Acquisition constants -------------------------------------------------
# 0.37 um/px is this dataset's native CODEX pixel size. We deliberately do NOT
# pre-resample to a common 0.5 mpp: each model documents its own target
# resolution and resamples internally from the mpp it is told (DeepCell Types
# takes `mpp` as an argument; KRONOS2's CORAL patch config encodes it in the
# patch slug). Resampling first would interpolate twice for those models and
# would also move the KRONOS2 run off the resolution its published benchmark
# was measured at. Handing every model identical pixels plus an identical,
# truthful `mpp` is the harmonisation; see CLAUDE.md.
MPP = 0.37
NUCLEAR_MARKER = "DAPI"
CELL_PATCH_SIZE = 64
SEED = 42

# The 18-marker phenotypic panel CORAL's tutorials 3-5 extract on. Fixing the
# panel matters more than it looks: it is the only way the mean-marker baseline
# and the foundation models are reading the same channels.
PANEL = [
    "dapi", "cd11b", "cd11c", "cd15", "cd163", "cd20", "cd206", "cd30",
    "cd31", "cd4", "cd56", "cd68", "cd7", "cd8", "cytokeratin", "foxp3",
    "mct", "podoplanin",
]

# --- Label set -------------------------------------------------------------
# The published benchmark's class list, in its label-ID order. Two curation
# decisions produce it, both inherited from CORAL tutorial 5:
#   - `Seg Artifact` is dropped: those are segmentation failures, so scoring
#     them measures the segmenter, not the encoder.
#   - `Cytotoxic CD8` is merged into `CD8`: ~380 cells against CD8's ~17k is
#     too rare to score, and holding it out would also strip those cells
#     from CD8.
# `Other` is kept. It is a catch-all with no consistent phenotype and it is the
# hardest class, which is exactly why dropping it would make the numbers
# incomparable with the published ones.
CLASS_NAMES = [
    "B", "CD4", "CD8", "DC", "Endothelial", "Epithelial", "Lymphatic",
    "M1", "M2", "Mast", "Monocyte", "NK", "Neutrophil", "Other",
    "TReg", "Tumor",
]
LABELS = np.arange(len(CLASS_NAMES))

DROP_LABELS = ("Seg Artifact",)
MERGE_LABELS = {"Cytotoxic CD8": "CD8"}

# --- Probe protocol --------------------------------------------------------
N_TRIALS = 15                 # Optuna trials per fold
C_RANGE = (1e-4, 1e2)         # log-uniform search bounds on C
MAX_ITER = 10000
MAX_CELLS_PER_CLASS = 2000    # equal training budget per encoder

# --- Few-shot protocol -----------------------------------------------------
# Support-set sizes for the label-efficiency sweep. Drawn from the training
# quadrants only, so a support cell is never a spatial neighbour of a test
# cell.
SHOT_SIZES = (10, 20, 100)

CELLS_PARQUET = RESULTS / "cells.parquet"


def assign_quadrants(
    x: np.ndarray, y: np.ndarray, width: int, height: int, band: int = CELL_PATCH_SIZE
) -> np.ndarray:
    """Label each cell with a spatial quadrant, or 0 for the guard band.

    Quartering the slide and holding out one quadrant at a time keeps train and
    test on physically separate tissue. Without that, spatial autocorrelation
    inflates the score: neighbouring cells share microenvironment and staining,
    and 64 px patches on centroids 20 px apart literally share pixels.

    The guard band closes the remaining seam. A cell sitting just left of the
    midline still has a patch reaching across it, so cells within ``band`` of
    either midline are dropped from every fold rather than assigned to one.

    Args:
        x: Cell centroid x coordinates, in level-0 pixels.
        y: Cell centroid y coordinates, in level-0 pixels.
        width: Slide width in level-0 pixels.
        height: Slide height in level-0 pixels.
        band: Guard-band half-width. Defaults to one patch width.

    Returns:
        Integer array of quadrant ids: 1..4 for TL, TR, BL, BR, and 0 for
        cells inside the guard band.
    """
    mid_x, mid_y = width // 2, height // 2
    left, right = x <= mid_x - band, x >= mid_x + band
    top, bottom = y <= mid_y - band, y >= mid_y + band
    return np.select(
        [left & top, right & top, left & bottom, right & bottom],
        [1, 2, 3, 4],
        default=0,
    )


def curate_labels(labels: pd.Series) -> pd.Series:
    """Apply the benchmark's label curation: drop artifacts, merge subtypes.

    Args:
        labels: Raw per-cell label strings.

    Returns:
        The same series with :data:`MERGE_LABELS` applied. Rows carrying a
        :data:`DROP_LABELS` value are left in place for the caller to filter,
        so the caller can apply the identical mask to its feature matrix.
    """
    return labels.replace(MERGE_LABELS)


def load_cells() -> pd.DataFrame:
    """Load the canonical cell table every model is scored on.

    Returns:
        DataFrame indexed by ``cell_id`` with columns ``label`` (string),
        ``code`` (integer class code into :data:`CLASS_NAMES`), ``x``, ``y``
        (centroids in level-0 pixels) and ``quadrant`` (1..4).

    Raises:
        FileNotFoundError: If ``prep_coral.py`` has not run yet.
    """
    if not CELLS_PARQUET.exists():
        raise FileNotFoundError(
            f"No canonical cell table at {CELLS_PARQUET}. "
            "Run src/prep_coral.py (env: sp-coral) first."
        )
    return pd.read_parquet(CELLS_PARQUET)


def open_slide() -> "object":
    """Open the ingested CORAL slide every pixel-reading model works from."""
    from coral import CoralSlide

    return CoralSlide.open(STORE)


def panel_indices(slide: "object", markers: list[str] | None = None) -> list[int]:
    """Resolve panel marker names to plane indices in the ingested slide.

    The one definition of *which planes are the panel*. Every model that reads
    pixels outside CORAL's own patch pipeline goes through this, so a model
    cannot silently end up on a different channel set — which is precisely how a
    panel drifts between models.

    Args:
        slide: An open :class:`coral.CoralSlide`.
        markers: Marker names to resolve. Defaults to the full :data:`PANEL`.

    Returns:
        Plane indices into ``slide.image``, positionally aligned to ``markers``.

    Raises:
        ValueError: If any requested marker is absent from the slide.
    """
    markers = list(PANEL if markers is None else markers)
    available = list(slide.markers)
    missing = [m for m in markers if m not in available]
    if missing:
        raise ValueError(f"panel markers absent from the slide: {missing}")
    return [available.index(m) for m in markers]


def load_mask() -> np.ndarray:
    """Load the published cell mask in level-0 pixel space.

    Read from the original TIFF rather than re-derived: CORAL's custom-mask
    import preserves cell ids, so these ids join directly onto the canonical
    cell table.
    """
    import tifffile

    return tifffile.imread(MASK_TIFF)


def load_panel_stack(
    markers: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the panel channels and the cell mask in level-0 pixel space.

    Channels come from the ingested CORAL slide rather than the raw TIFFs, so
    every model reads exactly the planes KRONOS2 read, in the same order.

    Args:
        markers: Marker names to load. Defaults to the full :data:`PANEL`.

    Returns:
        ``(raw, marker_names, mask)`` — ``raw`` is ``(C, H, W)`` float32,
        ``marker_names`` are CORAL's canonical names, ``mask`` is a 2D label
        image.
    """
    markers = list(PANEL if markers is None else markers)
    slide = open_slide()
    raw = np.asarray(slide.image[panel_indices(slide, markers)].values, dtype=np.float32)
    mask = load_mask()
    print(f"raw {raw.shape} {raw.dtype} | mask {mask.shape} "
          f"({len(np.unique(mask)) - 1:,} cells)")
    return raw, markers, mask


def align_features(cells: pd.DataFrame, feature_path: Path) -> np.ndarray:
    """Reindex a model's feature matrix onto the canonical cell order.

    Every model writes an ``.npz`` holding ``cell_id`` and ``features``; the
    natural row order differs between them (CORAL orders by mask label,
    Spatium by tokenisation order), so alignment is by id, never by position.

    Args:
        cells: The canonical table from :func:`load_cells`.
        feature_path: Path to a ``.npz`` with ``cell_id`` and ``features``.

    Returns:
        Float32 array of shape ``(len(cells), n_features)``.

    Raises:
        ValueError: If any canonical cell is missing from the feature file.
    """
    blob = np.load(feature_path)
    ids, feats = blob["cell_id"], blob["features"]
    lookup = pd.Series(np.arange(len(ids)), index=ids)
    pos = lookup.reindex(cells.index)
    if pos.isna().any():
        raise ValueError(
            f"{feature_path.name} is missing {int(pos.isna().sum()):,} of "
            f"{len(cells):,} canonical cells; it cannot be scored on the "
            "shared folds."
        )
    return np.asarray(feats, dtype=np.float32)[pos.to_numpy(dtype=int)]

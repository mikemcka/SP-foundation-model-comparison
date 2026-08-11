"""VirTues: marker-aware cell summary tokens from the whole tissue.

Runs in the `sp-virtues` env, on a GPU.

VirTues reads a *tissue*, not a cell. It tiles the whole slide, encodes each tile
with alternating spatial and marker attention, and emits one summary token per
8x8 patch; a cell's representation is the pixel-weighted average of the patch
tokens its segmentation mask overlaps. So unlike KRONOS2 and DeepCell Types it
never sees a cell in isolation — every cell token carries its neighbourhood, by
construction.

Markers are identified to the model by *protein sequence*: each channel is
embedded with ESM-2 from the canonical amino-acid sequence of its target, and
that vector is added to every patch token of that channel. The panel therefore
has to be resolved onto UniProt accessions before anything can run
(`common/virtues_markers.yaml`), and a channel whose target is not a protein —
DAPI — has nothing to embed and is dropped. VirTues sees 17 of the 18 markers.

Three things here differ from the other model scripts, all forced by the model:

* **The slide is resampled to 1.0 um/px.** This is the one place the repo's
  "hand every model identical pixels plus a truthful mpp" rule cannot hold.
  VirTues takes no `mpp` argument and does no internal resampling: its training
  corpus (spora) is stored at a fixed 1.0 um/px, so its 8 px patch *is* an 8 um
  patch. Feeding it this slide's native 0.37 mpp would silently present every
  structure at 2.7x the scale it was pretrained on. Resampling is what keeps it
  on-distribution, so it is the honest choice here even though it is a
  deviation; area-averaging down is also the mildest resampling in the repo.

* **Preprocessing is replicated rather than imported.** VirTues' own
  `MultiplexDataset._preprocess` defines it exactly — clip at the per-image 99th
  percentile, log1p, Gaussian blur, standardise — but reaching it through the
  shipped loader would mean converting this slide into the spora on-disk format
  first, which forks the mask and panel handling into a second implementation.
  The four steps are reproduced against that source instead, from this repo's
  single canonical slide and mask.

* **No native-usage row.** VirTues' documented cell-phenotyping mode *is* a
  linear probe on frozen cell tokens (`notebooks/2_demo_cell_phenotyping.ipynb`),
  so its native mode and the supervision-matched comparison are the same
  measurement. There is nothing extra to report, unlike DeepCell Types
  (zero-shot) or Spatium (few-shot).

Weights are the published `virtues-sp32` instance from the public
`bunnelab/virtues` Hub repo (CC BY-NC 4.0 — academic use only).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import protocol  # noqa: E402

VIRTUES_ROOT = protocol.REPO / "virtues"
BASE_CONFIG = VIRTUES_ROOT / "configs" / "base_config.yaml"
MARKER_YAML = Path(__file__).resolve().parent / "common" / "virtues_markers.yaml"

WORK = protocol.RESULTS / "virtues"
FASTA_DIR = WORK / "fastas"
EMBED_ROOT = WORK / "marker_embeddings"
CHECKPOINT_DIR = WORK / "checkpoints"

# The ESM-2 instance the published VirTues weights were trained with. Its 640-d
# output is what `prior_embedding_encoder` expects, so this is a compatibility
# pin, not a quality choice.
ESM_MODEL = "esm2_t30_150M_UR50D"

HF_REPO = "bunnelab/virtues"
HF_WEIGHTS = "virtues-sp32/model.safetensors"

# The resolution VirTues' training corpus is stored at; see the module docstring.
VIRTUES_MPP = 1.0
# Sliding-window geometry. tile/patch come from configs/base_config.yaml; the
# stride and the zero-pad are `compute_cell_tokens`' and the demo notebook's
# defaults. Stride 42 over a 128 px tile means every pixel is encoded ~9 times
# in different contexts, which is what makes the per-cell average stable.
TILE_SIZE = 128
PATCH_SIZE = 8
STRIDE = 42
PAD = 120
# Upper clipping quantile, per channel, from the standardisation pipeline the
# published weights were trained under ('quantile_clipping_log1p/uq_0.99_image').
UPPER_QUANTILE = 0.99


def run(cmd: list[str]) -> None:
    """Run one of VirTues' own utility scripts, echoing it first."""
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_panel() -> tuple[list[str], list[str]]:
    """Resolve the panel onto UniProt accessions, dropping non-protein channels.

    This is VirTues' marker-harmonisation step, and it is load-bearing: an
    accession is the *only* handle the model has on a channel's identity.

    Returns:
        ``(markers, uniprot_ids)`` — the panel markers VirTues can see, and
        their accessions, positionally aligned.

    Raises:
        ValueError: If a panel marker has no entry at all (as opposed to an
            explicit ``null``), which means the panel changed without the map
            being revisited.
    """
    mapping = yaml.safe_load(MARKER_YAML.read_text())["uniprot"]
    unlisted = [m for m in protocol.PANEL if m not in mapping]
    if unlisted:
        raise ValueError(
            f"{unlisted} have no entry in {MARKER_YAML.name}. Add an accession, "
            "or `null` if the target is not a protein — the decision has to be "
            "explicit."
        )

    markers = [m for m in protocol.PANEL if mapping[m]]
    dropped = [m for m in protocol.PANEL if not mapping[m]]
    print("\nmarker resolution (CORAL -> UniProt):")
    for m in protocol.PANEL:
        print(f"  {m:<14} -> {mapping[m] or 'DROPPED (not a protein)'}")
    print(f"\nVirTues sees {len(markers)} of {len(protocol.PANEL)} panel markers"
          + (f"; dropped {dropped}" if dropped else ""))
    return markers, [mapping[m] for m in markers]


def ensure_marker_embeddings(uniprot_ids: list[str], *, device: str) -> Path:
    """Fetch each target's canonical sequence and embed it with ESM-2.

    Both steps are VirTues' own scripts, unmodified, so the embeddings are
    produced exactly as the ones the model was trained against. Needs internet;
    the GPU stage does not, which is why they are separate stages.

    Args:
        uniprot_ids: Accessions to embed.
        device: Torch device for ESM-2. CPU is fine and is the safe choice for
            long sequences.

    Returns:
        The directory of ``[accession].pt`` embeddings.

    Raises:
        RuntimeError: If any accession fails to download or embed —
            ``download_fastas`` only prints on a failed fetch, and a missing
            marker would otherwise show up as a silently weaker model.
    """
    embed_dir = EMBED_ROOT / ESM_MODEL
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    ids_csv = WORK / "panel_uniprot.csv"
    pd.DataFrame({"protein_id": uniprot_ids}).to_csv(ids_csv, index=False)

    run([sys.executable, "-m", "virtues.utils.download_fastas",
         "--output_dir", str(FASTA_DIR), "--input", str(ids_csv),
         "--id_column", "protein_id"])
    absent = [u for u in uniprot_ids if not (FASTA_DIR / f"{u}.fasta").exists()]
    if absent:
        raise RuntimeError(
            f"UniProt returned no sequence for {absent}. Check the accessions in "
            f"{MARKER_YAML.name} — a wrong one fails silently at download time."
        )

    run([sys.executable, "-m", "virtues.utils.compute_esm_embeddings",
         "--input_dir", str(FASTA_DIR), "--output_dir", str(EMBED_ROOT),
         "--model", ESM_MODEL, "--device", device])

    # The loader takes *every* .pt in the directory and orders them by filename,
    # so a stale embedding from an earlier panel would shift every marker index
    # by one without any error. Check the set matches exactly.
    on_disk = sorted(p.stem for p in embed_dir.glob("*.pt"))
    if on_disk != sorted(uniprot_ids):
        raise RuntimeError(
            f"{embed_dir} holds {on_disk}, expected {sorted(uniprot_ids)}. "
            "Delete the directory and re-run the markers stage."
        )
    print(f"\n{len(on_disk)} marker embeddings ready -> {embed_dir}")
    return embed_dir


def build_model(embed_dir: Path, uniprot_ids: list[str], *, device: str):
    """Instantiate VirTues with this panel's marker embeddings and load weights.

    Every architecture argument comes from VirTues' shipped
    ``configs/base_config.yaml``, which is the configuration the released
    weights were trained under. The marker embeddings are passed as a
    non-persistent buffer, so they are *not* part of the checkpoint: each
    dataset supplies its own panel's embeddings and the same weights index into
    them.

    Returns:
        ``(model, marker_indices)`` — the model in eval mode on ``device``, and
        the row of the embedding matrix each panel channel maps to.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.utils.utils import load_marker_embedding_dict, load_marker_embeddings

    conf = OmegaConf.load(BASE_CONFIG)
    embeddings = load_marker_embeddings(str(embed_dir))
    lookup = load_marker_embedding_dict(str(embed_dir))
    marker_indices = torch.tensor([lookup[u] for u in uniprot_ids], dtype=torch.long)

    model = MultiplexVirtues(
        use_default_config=False,
        custom_config=None,
        prior_bias_embeddings=embeddings,
        prior_bias_embedding_type="esm",
        prior_bias_embedding_fusion_type="add",
        patch_size=conf.model.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        decoder_pattern=conf.model.decoder_pattern,
        num_decoder_heads=conf.model.num_decoder_heads,
        num_hidden_layers=conf.model.num_decoder_hidden_layers,
        positional_embedding_type=conf.model.positional_embedding_type,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder,
        verbose=False,
    )
    weights = hf_hub_download(
        repo_id=HF_REPO, filename=HF_WEIGHTS, local_dir=str(CHECKPOINT_DIR)
    )
    model.load_state_dict(load_file(weights, device="cpu"))
    print(f"loaded {HF_WEIGHTS} ({embeddings.shape[1]}-d marker embeddings, "
          f"{conf.model.model_dim}-d tokens)")
    return model.to(device).eval(), marker_indices.to(device)


def rescale(raw: np.ndarray, mask: np.ndarray, cells: pd.DataFrame):
    """Resample the slide and mask from 0.37 to 1.0 um/px.

    Intensities are area-averaged, which is the right reduction for a downscale:
    it is the mean of the pixels a target pixel covers, so it neither invents
    signal nor drops isolated bright pixels the way point sampling would.

    The mask is *point sampled* instead — a label image has no meaningful
    average. At 2.7x down, a handful of the smallest cells can lose every pixel
    they had; those are restored at their own centroid, so no cell in the
    canonical table can go missing from the feature matrix (which would be an
    unscorable hole, not a small error).

    Args:
        raw: ``(C, H, W)`` float32 intensities at :data:`protocol.MPP`.
        mask: ``(H, W)`` label image at :data:`protocol.MPP`.
        cells: The canonical cell table, for the centroid restore.

    Returns:
        ``(image, mask)`` at :data:`VIRTUES_MPP`.
    """
    import torch

    scale = protocol.MPP / VIRTUES_MPP
    h, w = mask.shape
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    image = np.empty((len(raw), new_h, new_w), dtype=np.float32)
    for c in range(len(raw)):
        plane = torch.from_numpy(raw[c])[None, None]
        image[c] = torch.nn.functional.interpolate(
            plane, size=(new_h, new_w), mode="area"
        )[0, 0].numpy()

    rows = np.minimum(((np.arange(new_h) + 0.5) / scale).astype(int), h - 1)
    cols = np.minimum(((np.arange(new_w) + 0.5) / scale).astype(int), w - 1)
    small = mask[rows[:, None], cols[None, :]]

    lost = np.setdiff1d(cells.index.to_numpy(), np.unique(small), assume_unique=False)
    if len(lost):
        rows_ = np.clip((cells.loc[lost, "y"].to_numpy() * scale).astype(int), 0, new_h - 1)
        cols_ = np.clip((cells.loc[lost, "x"].to_numpy() * scale).astype(int), 0, new_w - 1)
        small[rows_, cols_] = lost
    print(f"resampled {protocol.MPP} -> {VIRTUES_MPP} mpp: {(h, w)} -> "
          f"{(new_h, new_w)} px | restored {len(lost):,} cells at their centroid")
    return image, small


def standardize(image: np.ndarray, *, blur: bool):
    """Apply VirTues' image preprocessing, then zero-pad for the sliding window.

    Reproduces ``virtues.data.multiplex_dataset.MultiplexDataset._preprocess``
    step for step: clip to ``[0, q99]`` per channel, log1p, Gaussian blur
    (3x3, sigma 1), then standardise by the channel's log-scale mean and std.

    Statistics come from this slide, which is what spora's ``uq_0.99_image``
    pipeline does for the quantile. Its mean and std are corpus-level; with a
    single slide there is nothing else to compute them from, and using this
    slide's own is the closer approximation of the two available.

    Statistics are computed *before* padding so the zeros cannot drag them,
    which is the order the demo notebook gets for free by loading precomputed
    stats. The blur runs before padding too, for a smaller reason: on the padded
    image it would mix tissue with the zero moat and leave a darkened one-pixel
    ring around the slide. Blurring first leaves the border reflected, which is
    what the training path sees.

    Args:
        image: ``(C, H, W)`` float32 intensities at :data:`VIRTUES_MPP`.
        blur: Apply the Gaussian blur. On by default because the pretraining
            path applies it; the phenotyping notebook standardises without it,
            and ``--no-blur`` runs that reading. See CLAUDE.md.

    Returns:
        A ``(C, H + 2*PAD, W + 2*PAD)`` float32 torch tensor.
    """
    import torch
    from torchvision.transforms import v2

    out = torch.empty(
        (len(image), image.shape[1] + 2 * PAD, image.shape[2] + 2 * PAD),
        dtype=torch.float32,
    )
    gaussian = v2.GaussianBlur(kernel_size=3, sigma=1.0)
    for c in range(len(image)):
        upper = float(np.quantile(image[c], UPPER_QUANTILE))
        plane = torch.log1p(torch.from_numpy(image[c]).clamp(min=0, max=upper))
        mean, std = plane.mean(), plane.std()
        if blur:
            plane = gaussian(plane[None])[0]
        plane = torch.nn.functional.pad(plane[None], (PAD, PAD, PAD, PAD))
        out[c] = (plane[0] - mean) / (std + 1e-9)
    print(f"standardised {tuple(out.shape)}"
          + ("" if blur else " (blur skipped)"))
    return out


def cell_tokens(
    model,
    image,
    marker_indices,
    mask: np.ndarray,
    cells: pd.DataFrame,
    *,
    device: str,
    stride: int,
    chunk_size: int,
) -> np.ndarray:
    """Encode the slide tile by tile and pool patch tokens into cell tokens.

    Same computation as ``virtues.utils.cell_tokens.compute_cell_tokens`` — the
    same crop grid (its own ``_get_uniform_crops`` decides it), the same patch
    tokens, the same pixel-overlap-weighted average per cell — but accumulated
    into a dense array as it goes instead of collected in per-cell Python lists.
    On this slide the shipped version would hold every crop's tokens plus a few
    million single-token tensors in memory at once, which is tens of GB; a
    ~139k x 512 running sum is 285 MB.

    Two consequences of doing the bookkeeping directly, both exact rather than
    approximations:

    * Crops containing no canonical cell are never encoded. A crop only ever
      contributes to cells inside it, so skipping them changes no output — and
      on a slide with background margins it is most of the compute.
    * Cells outside the canonical table (unlabelled, artifact, guard band) are
      not accumulated at all.

    Args:
        model: The loaded VirTues model.
        image: Padded, standardised ``(C, H, W)`` tensor.
        marker_indices: Marker embedding row per channel, on ``device``.
        mask: Padded label image, same ``(H, W)`` as ``image``.
        cells: The canonical cell table; output rows follow its order.
        device: Torch device (CUDA — VirTues' attention is FlashAttention).
        stride: Sliding-window stride in pixels.
        chunk_size: Crops per forward pass.

    Returns:
        Float32 ``(len(cells), model_dim)`` cell summary tokens.
    """
    import torch

    from virtues.utils.cell_tokens import _get_uniform_crops

    grid = TILE_SIZE // PATCH_SIZE  # patches per tile edge
    n_cells = len(cells)

    # Dense id -> row lookup; -1 for every mask id that is not scored.
    row_of_id = np.full(int(max(mask.max(), cells.index.max())) + 1, -1, dtype=np.int64)
    row_of_id[cells.index.to_numpy()] = np.arange(n_cells)
    patch_of_pixel = np.repeat(np.arange(grid * grid), PATCH_SIZE * PATCH_SIZE)

    _, indices = _get_uniform_crops(image, stride, tile_size=TILE_SIZE)
    print(f"\n{len(indices):,} crops of {TILE_SIZE}px at stride {stride}", flush=True)

    sums = torch.zeros((n_cells, model.encoder.model_dim), device=device)
    weights = torch.zeros(n_cells, device=device)
    encoded = 0

    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]

        # Mask side first: it decides which crops are worth a forward pass.
        crops, keys = [], []
        for row, col in chunk:
            block = mask[row : row + TILE_SIZE, col : col + TILE_SIZE]
            rows_ = row_of_id[
                block.reshape(grid, PATCH_SIZE, grid, PATCH_SIZE)
                .transpose(0, 2, 1, 3)
                .reshape(-1)
            ]
            keep = rows_ >= 0
            if not keep.any():
                continue
            patches = patch_of_pixel[keep] + len(crops) * grid * grid
            keys.append(patches.astype(np.int64) * n_cells + rows_[keep])
            crops.append(image[:, row : row + TILE_SIZE, col : col + TILE_SIZE])
        if not crops:
            continue

        batch = [c.to(device, non_blocking=True).contiguous() for c in crops]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            out = model.encoder.forward_list(
                batch, [marker_indices] * len(batch), multiplex_mask=None
            )
        tokens = torch.stack(out.patch_summary_tokens).float().reshape(-1, sums.shape[1])

        # One (patch, cell) pair per unique key, weighted by overlapping pixels.
        pair, overlap = np.unique(np.concatenate(keys), return_counts=True)
        rows_t = torch.as_tensor(pair % n_cells, device=device)
        cols_t = torch.as_tensor(pair // n_cells, device=device)
        w = torch.as_tensor(overlap, dtype=torch.float32, device=device)
        sums.index_add_(0, rows_t, tokens[cols_t] * w[:, None])
        weights.index_add_(0, rows_t, w)

        encoded += len(crops)
        if (start // chunk_size) % 50 == 0:
            print(f"  {start + len(chunk):>7,}/{len(indices):,} crops "
                  f"({encoded:,} encoded)", flush=True)

    empty = int((weights == 0).sum())
    if empty:
        raise ValueError(
            f"{empty:,} canonical cells got no patch token. The mask and the "
            "image are out of register — check the resampling."
        )
    print(f"encoded {encoded:,} of {len(indices):,} crops "
          f"({1 - encoded / len(indices):.0%} skipped as cell-free)")
    return (sums / weights[:, None]).cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--esm-device",
        default="cpu",
        help="Device for the one-off ESM-2 marker embeddings. CPU is safe for "
        "long sequences and takes under a minute for 17 proteins.",
    )
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--no-blur",
        dest="blur",
        action="store_false",
        help="Skip the Gaussian blur in preprocessing (see CLAUDE.md).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "markers", "tokens"),
        default="all",
        help="markers needs internet and no GPU; tokens needs a GPU and no "
        "internet beyond the checkpoint download.",
    )
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    markers, uniprot_ids = resolve_panel()

    if args.stage in ("all", "markers"):
        embed_dir = ensure_marker_embeddings(uniprot_ids, device=args.esm_device)
    else:
        embed_dir = EMBED_ROOT / ESM_MODEL
    if args.stage == "markers":
        return

    if not args.device.startswith("cuda"):
        raise RuntimeError(
            f"--device {args.device}: VirTues' attention blocks are FlashAttention, "
            "which is CUDA-only. The tokens stage needs a GPU."
        )

    cells = protocol.load_cells()
    model, marker_indices = build_model(embed_dir, uniprot_ids, device=args.device)

    raw, _, mask = protocol.load_panel_stack(markers)
    image, mask = rescale(raw, mask, cells)
    del raw
    image = standardize(image, blur=args.blur)
    mask = np.pad(mask, PAD)

    features = cell_tokens(
        model, image, marker_indices, mask, cells,
        device=args.device, stride=args.stride, chunk_size=args.chunk_size,
    )
    out = protocol.RESULTS / "features" / "virtues.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cell_id=cells.index.to_numpy(), features=features)
    print(f"wrote VirTues cell tokens {features.shape} -> {out}")


if __name__ == "__main__":
    main()

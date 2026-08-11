"""DeepCell Types: zero-shot cell-type calls plus frozen CLS embeddings.

Runs in the `sp-dct` env, on a GPU.

DeepCell Types is used two ways here, and they answer different questions:

* **Zero-shot** (its headline capability) — it has never seen this slide or its
  label set, and predicts from its own 51-type vocabulary by matching the
  panel's channels against a packaged marker registry. Scored after mapping its
  vocabulary onto the 16 cHL classes (`dct_class_map.yaml`). This measures the
  model *as shipped*, which is the honest way to use it, but it is scored
  against a label space it was never told about — a mapping miss costs it a
  cell just as much as a real error does.

* **Frozen CLS embedding** — the same backbone, but read as a representation
  and handed to the identical linear probe the other encoders get. This is the
  supervision-matched comparison; it isolates representation quality from
  vocabulary alignment.

Both run off the same 18-marker panel, the same slide pixels and the same cell
mask as KRONOS2, so the only thing that differs between models is the encoder.

The embedding pass reaches into `deepcell_types.predict`'s private helpers
(`_resolve_model_file`, `_build_model`, `PatchDataset`). There is no public API
for embeddings, and reimplementing the patch construction would risk feeding the
model something subtly different from what `predict()` feeds it — which would
make the two DCT rows incomparable to each other.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import protocol  # noqa: E402

CLASS_MAP_YAML = Path(__file__).resolve().parent / "common" / "dct_class_map.yaml"

# Marker aliases DeepCell Types' own resolver does not cover. Both are genuine
# name differences for the same antigen, not substitutions:
#
#   dapi -> dsDNA    DeepCell Types' registry carries the nuclear channel under
#                    the MIBI-era name `dsDNA`; DAPI is the CODEX stain for the
#                    same target (nuclear DNA). Without the alias the model
#                    would run with no nuclear channel at all, which costs it
#                    the morphology signal every other encoder here gets.
#   mct  -> Tryptase This dataset abbreviates mast cell tryptase; the registry
#                    spells it out.
#
# `cd30` is deliberately absent: the registry has no CD30 / TNFRSF8 / Ki-1
# entry, so DeepCell Types cannot see the marker that *defines* the Tumor
# (Hodgkin Reed-Sternberg) class in this dataset. That is a real capability
# limit, not a naming problem, so it is left unresolved and reported rather
# than papered over with a stand-in channel.
MARKER_ALIASES = {
    "dapi": "dsDNA",
    "mct": "Tryptase",
}


def check_credentials() -> None:
    """Fail early if the gated checkpoint download has no token."""
    if not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
        raise RuntimeError(
            "DEEPCELL_ACCESS_TOKEN is not set; the checkpoint download is gated "
            "behind users.deepcell.org. Run: source env/secrets.env"
        )


def resolve_markers(names: list[str]) -> list[str]:
    """Map CORAL's canonical marker names onto DeepCell Types' registry.

    This is the marker-vocabulary harmonisation step: CORAL lower-cases
    (``cd11b``, ``cytokeratin``), DeepCell Types has its own canonical spellings
    and alias table. :data:`MARKER_ALIASES` covers the two names its resolver
    misses. An unresolvable marker is passed through unchanged — the model drops
    out-of-vocabulary channels itself — but is reported, because a silently
    dropped channel is a silently weakened model.

    Args:
        names: CORAL canonical marker names.

    Returns:
        Names in DeepCell Types' vocabulary, positionally aligned to ``names``.
    """
    from deepcell_types.utils import resolve_supported_marker

    resolved, unmapped = [], []
    for name in names:
        canonical = resolve_supported_marker(MARKER_ALIASES.get(name, name))
        if canonical is None:
            unmapped.append(name)
            resolved.append(name)
        else:
            resolved.append(canonical)

    print("\nmarker resolution (CORAL -> DeepCell Types):")
    for src, dst in zip(names, resolved):
        flag = "  UNRESOLVED" if src in unmapped else ""
        print(f"  {src:<14} -> {dst}{flag}")
    if unmapped:
        print(f"\n{len(unmapped)} of {len(names)} markers are out of vocabulary "
              "and will be ignored by the model.")
    return resolved


def run_zero_shot(
    raw: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    *,
    model_path: str,
    device: str,
    batch_size: int,
) -> None:
    """Predict cell types with the shipped checkpoint and save the calls.

    Saves both the raw vocabulary labels and the per-cell probabilities over
    DeepCell Types' own classes, so the mapping onto the cHL label space stays a
    separate, inspectable step rather than being baked into the inference run.
    """
    from deepcell_types import predict
    from deepcell_types.config import DCTConfig

    result = predict(
        raw,
        mask,
        channel_names,
        protocol.MPP,
        model_name=model_path,
        device=device,
        batch_size=batch_size,
        return_probabilities=True,
    )

    dct_classes = [
        name for name, _ in sorted(DCTConfig().ct2idx.items(), key=lambda kv: kv[1])
    ]
    out_dir = protocol.RESULTS / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"cell_id": result.cell_indices, "dct_label": result.cell_types}
    ).to_csv(out_dir / "dct_zeroshot_raw.csv", index=False)
    np.savez(
        out_dir / "dct_zeroshot_probs.npz",
        cell_id=np.asarray(result.cell_indices),
        probabilities=np.asarray(result.probabilities, dtype=np.float32),
        classes=np.asarray(dct_classes),
    )
    print(f"\nzero-shot: {len(result.cell_types):,} cells over "
          f"{len(dct_classes)} DeepCell Types classes -> {out_dir}")
    print(pd.Series(result.cell_types).value_counts().head(25).to_string())


def run_embeddings(
    raw: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    *,
    model_path: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Capture the frozen CLS embedding for every cell.

    Mirrors ``predict()``'s inference loop exactly, reading ``cls_embedding``
    off the model output instead of the logits, so the embeddings correspond
    cell-for-cell with the zero-shot calls.
    """
    import torch
    from torch.utils.data import DataLoader

    from deepcell_types.config import DCTConfig
    from deepcell_types.dataset import PatchDataset
    from deepcell_types.predict import (
        _build_model,
        _resolve_model_file,
        _torch_load_weights,
    )

    dev = torch.device(device)
    checkpoint = _torch_load_weights(_resolve_model_file(model_path), dev)
    dct_config = DCTConfig()
    model = _build_model(checkpoint, dct_config, dev)
    model.eval()

    dataset = PatchDataset(raw, mask, channel_names, protocol.MPP, dct_config)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    embeddings, cell_ids = [], []
    with torch.no_grad():
        for sample, spatial_context, ch_idx, attn_mask, cell_index in loader:
            out = model(
                sample.to(dev),
                spatial_context.to(dev),
                ch_idx.to(dev),
                attn_mask.to(dev),
            )
            embeddings.append(out.cls_embedding.cpu().numpy())
            cell_ids.append(cell_index.numpy())

    features = np.concatenate(embeddings).astype(np.float32)
    ids = np.concatenate(cell_ids)
    order = np.argsort(ids, kind="stable")

    out = protocol.RESULTS / "features" / "deepcell_types.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cell_id=ids[order], features=features[order])
    print(f"wrote DeepCell Types CLS embeddings {features.shape} -> {out}")


def map_to_benchmark() -> None:
    """Translate DeepCell Types' vocabulary onto the 16 cHL classes.

    Kept separate from inference so the mapping is a reviewable artifact. Any
    DCT class with no cHL counterpart maps to ``-1``, which
    :func:`common.probe.score_predictions` scores as an error rather than
    dropping — a model that answers "Fibroblast" on a slide whose label set has
    no fibroblast class did get that cell wrong, from the benchmark's point of
    view.
    """
    mapping = yaml.safe_load(CLASS_MAP_YAML.read_text())["dct_to_chl"]
    raw = pd.read_csv(protocol.RESULTS / "predictions" / "dct_zeroshot_raw.csv")

    unseen = sorted(set(raw["dct_label"]) - set(mapping))
    if unseen:
        raise ValueError(
            f"{len(unseen)} predicted DeepCell Types classes are absent from "
            f"{CLASS_MAP_YAML.name}: {unseen}. Add them (use `null` for classes "
            "with no cHL counterpart) so the mapping stays explicit."
        )

    code = {name: i for i, name in enumerate(protocol.CLASS_NAMES)}
    raw["chl_label"] = raw["dct_label"].map(mapping)
    raw["code"] = raw["chl_label"].map(lambda v: code.get(v, -1)).astype(int)

    out = protocol.RESULTS / "predictions" / "dct_zeroshot.csv"
    raw.to_csv(out, index=False)
    n_unmapped = int((raw["code"] < 0).sum())
    print(f"\nmapped {len(raw):,} calls onto the cHL label space -> {out}")
    print(f"{n_unmapped:,} cells ({n_unmapped / len(raw):.1%}) fell outside it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--stage",
        choices=("all", "zeroshot", "embeddings", "map"),
        default="all",
        help="Run one stage only (the two GPU stages are independent).",
    )
    args = parser.parse_args()

    if args.stage == "map":
        map_to_benchmark()
        return

    check_credentials()
    from deepcell_types.utils import download_model

    model_path = str(download_model())
    print(f"checkpoint: {model_path}")

    raw, names, mask = protocol.load_panel_stack()
    channel_names = resolve_markers(names)

    if args.stage in ("all", "zeroshot"):
        run_zero_shot(
            raw, mask, channel_names,
            model_path=model_path, device=args.device, batch_size=args.batch_size,
        )
    if args.stage in ("all", "embeddings"):
        run_embeddings(
            raw, mask, channel_names,
            model_path=model_path, device=args.device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    if args.stage == "all":
        map_to_benchmark()


if __name__ == "__main__":
    main()

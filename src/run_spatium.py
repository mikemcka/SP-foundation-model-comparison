"""Spatium: rank-token embeddings and few-shot prototype fine-tuning.

Runs in the `sp-spatium` env, on a GPU.

Spatium is not a vision model. It reads a cell as a *sentence of proteins*: take
the cell's mean intensity per marker, sort the markers from most to least
abundant, and feed that rank order — plus four metadata tokens (technology,
tissue, health status, disease) — to a transformer. Two consequences shape this
script:

* **The expression matrix must be the same one the mean-marker baseline uses.**
  It is read straight from `results/features/mean_marker.npz` rather than
  recomputed, so Spatium and the baseline are provably reading identical numbers
  and any difference is the model, not the measurement.

* **Per-marker z-scoring is not optional.** Only the rank *order* survives
  tokenisation, so on raw intensities the ranking is dominated by which
  antibodies are globally bright — nearly the same order for every cell, which
  destroys the signal. Z-scoring each marker across the dataset makes the
  ranking say "which markers are high *for this cell*", which is the question
  the model was pretrained on.

Two usage modes, matching the other two models:

* **Frozen CLS embedding** + the shared linear probe — the supervision-matched
  comparison.
* **Few-shot prototype fine-tuning** — Spatium's native few-shot mode, using its
  own `spaProFormerFinetune` head and losses, trained on support sets drawn from
  the fold's training quadrants only.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spatium_shim  # noqa: E402
from common import protocol  # noqa: E402

spatium_shim.setup()

SPATIUM_ROOT = Path(__file__).resolve().parents[1] / "Spatium"
CHECKPOINT = SPATIUM_ROOT / "final.ckpt"
WORK = protocol.RESULTS / "spatium"

# Metadata tokens. Spatium's controlled vocabulary has no lymphoma entry, so
# `Disease_type` pads — the model is told the tissue is a diseased lymph node
# imaged by CODEX, but not which disease. That is a real limit of the shipped
# vocabulary on this dataset, not a modelling choice, and is reported as such.
TECHNOLOGY = "CODEX"
TISSUE_TYPE = "Lymph_node"
HEALTH_STATUS = "Disease"
DISEASE_TYPE = "Hodgkin_lymphoma"  # not in side_maps -> PAD


def build_anndata() -> "object":
    """Assemble the per-cell expression matrix Spatium tokenises.

    Returns:
        An AnnData of shape ``(n_cells, n_panel_markers)`` carrying z-scored
        expression and the four metadata columns, with ``obs_names`` set to the
        canonical ``cell_id``.
    """
    import anndata as ad
    from constants import DataSchema

    cells = protocol.load_cells()
    X = protocol.align_features(
        cells, protocol.RESULTS / "features" / "mean_marker.npz"
    )
    print(f"expression matrix: {X.shape} (cells x panel markers)")

    # Dataset-level z-score per marker. Guard against a constant channel, which
    # would otherwise divide by zero and emit NaN — the tokeniser reads NaN as
    # "not measured" and would silently drop that marker from every cell.
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    dead = sigma == 0
    if dead.any():
        names = [m for m, d in zip(protocol.PANEL, dead) if d]
        warnings.warn(f"constant markers left un-scaled: {names}", stacklevel=2)
        sigma = np.where(dead, 1.0, sigma)
    Xz = (X - mu) / sigma

    adata = ad.AnnData(
        X=np.asarray(Xz, dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_type": cells["label"].to_numpy()},
            index=cells.index.astype(str),
        ),
        var=pd.DataFrame(index=pd.Index(protocol.PANEL, name="marker")),
    )
    DataSchema().set(
        adata,
        Technology=TECHNOLOGY,
        Tissue_type=TISSUE_TYPE,
        Health_status=HEALTH_STATUS,
        Disease_type=DISEASE_TYPE,
    )
    return adata


def tokenize(adata: "object") -> np.ndarray:
    """Rank-tokenise the expression matrix with Spatium's own tokeniser.

    Uses `FineTuneTokenizer` unmodified, including its protein-name
    standardisation table — that table is what turns CODEX antibody names into
    Spatium's gene-symbol vocabulary (``cd20`` -> ``MS4A1``, ``cd11b`` ->
    ``ITGAM``, ``cytokeratin`` -> ``KRT``), i.e. the marker-harmonisation step
    of the protocol.

    Args:
        adata: The AnnData from :func:`build_anndata`.

    Returns:
        Int64 token matrix of shape ``(n_cells, context_length)``.
    """
    from constants import PROTEIN_TOKEN_BASE, all_proteins, protein_id_map, side_maps
    from tokenizer import FineTuneTokenizer

    WORK.mkdir(parents=True, exist_ok=True)
    h5ad_path = WORK / "chl_codex.h5ad"
    adata.write_h5ad(h5ad_path)  # the tokeniser reads a path, not an object

    tokenizer = FineTuneTokenizer(
        all_proteins=all_proteins,
        protein_id_map=protein_id_map,
        side_maps=side_maps,
        protein_token_base=PROTEIN_TOKEN_BASE,
    )
    tokenizer.collect_data([str(h5ad_path)], cell_type_col="cell_type")
    tokens = tokenizer.tokenize()[0].astype(np.int64)

    kept = int((tokens[0, 5:] > 0).sum())
    print(f"\ntokens: {tokens.shape} | {kept} of {len(protocol.PANEL)} panel "
          "markers resolved into Spatium's vocabulary")
    np.save(WORK / "tokens.npy", tokens)
    return tokens


def embed(tokens: np.ndarray, *, device: str, batch_size: int) -> None:
    """Read the frozen [CLS] embedding for every cell.

    The pretrained encoder is used exactly as `spaProFormer.forward` defines it,
    with Spatium's own padding convention (``token == 0``), and the CLS position
    is taken from the transformer output.
    """
    import torch
    from model import spaProFormer

    cells = protocol.load_cells()
    model = spaProFormer.load_from_checkpoint(str(CHECKPOINT), map_location=device)
    model.eval().to(device)

    out = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            x = torch.as_tensor(tokens[start : start + batch_size], device=device)
            hidden = model(x, (x == 0))["transformer_output"]
            out.append(hidden[:, 0].float().cpu().numpy())

    features = np.concatenate(out).astype(np.float32)
    path = protocol.RESULTS / "features" / "spatium.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cell_id=cells.index.to_numpy(), features=features)
    print(f"wrote Spatium CLS embeddings {features.shape} -> {path}")


def _write_zarr(rows: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Write Spatium's few-shot input layout: tokens with the label appended."""
    import zarr

    payload = np.concatenate([rows, labels.reshape(-1, 1)], axis=1).astype(np.int64)
    store = zarr.open(str(path), mode="w", shape=payload.shape, dtype="int64")
    store[:] = payload


def few_shot(tokens: np.ndarray, *, device: str, max_epochs: int) -> None:
    """Fine-tune Spatium's prototype head on few-shot support sets.

    Spatium's own `spaProFormerFinetune` (task ``Prototype_classification``),
    losses and dataset classes are used unchanged; only the data plumbing and
    the trainer are local, because the shipped `few_shot_finetune.py` hardcodes
    the authors' paths, a Weights & Biases logger and multi-GPU DDP.

    Support cells come from the fold's *training* quadrants, so — exactly as
    for the linear probe — no support cell is a spatial neighbour of a scored
    cell. Every shot size is run on every fold.
    """
    import pytorch_lightning as pl
    import torch
    from dataloader import MerlinFewShotDataModule
    from fine_tune import spaProFormerFinetune
    from model import spaProFormer

    from common import probe

    cells = protocol.load_cells()
    y = cells["code"].to_numpy()
    n_classes = len(protocol.CLASS_NAMES)
    rows = []

    for n_shot in protocol.SHOT_SIZES:
        oof = np.full(len(cells), -1, dtype=np.int64)
        for fold_id in (1, 2, 3, 4):
            support = probe.draw_support_set(cells, fold_id, n_shot)
            _, _, test_idx = probe.split_fold(
                cells["quadrant"].to_numpy(), fold_id, random_state=protocol.SEED
            )

            fold_dir = WORK / f"fewshot_{n_shot}shot_fold{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            _write_zarr(tokens[support], y[support], fold_dir / "support")
            _write_zarr(tokens[test_idx], y[test_idx], fold_dir / "query")

            dm = MerlinFewShotDataModule(
                support_path=str(fold_dir / "support"),
                query_path=str(fold_dir / "query"),
                batch_size=min(256, len(support)),
                num_workers=2,
                task="Prototype_classification",
            )
            dm.setup()

            pl.seed_everything(protocol.SEED, workers=True)
            pretrained = spaProFormer.load_from_checkpoint(
                str(CHECKPOINT), map_location="cpu"
            )
            model = spaProFormerFinetune(
                pretrained_model=pretrained,
                num_cell_types=n_classes,
                drop_out=0.2,
                task="Prototype_classification",
                lr=1e-5,
                finetune_mode="full",
            )
            trainer = pl.Trainer(
                accelerator="gpu" if device.startswith("cuda") else "cpu",
                devices=1,
                max_epochs=max_epochs,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                log_every_n_steps=1,
                # No validation during the fit. Spatium's FewShotTokenDataset
                # reads the zarr one row per __getitem__, so validating on the
                # whole query quadrant each epoch is ~35k individual reads off a
                # network filesystem, x30 epochs x12 fits — it dominates the run
                # while buying nothing, because the scored predictions come from
                # the in-memory pass below, not from Lightning's val loop.
                limit_val_batches=0,
                num_sanity_val_steps=0,
            )
            trainer.fit(model, datamodule=dm)

            # Prototype classification: cosine similarity between the cell's
            # CLS embedding and each learned class prototype.
            model.eval().to(device)
            preds = []
            with torch.no_grad():
                for start in range(0, len(test_idx), 512):
                    x = torch.as_tensor(
                        tokens[test_idx[start : start + 512]], device=device
                    )
                    emb = torch.nn.functional.normalize(model(x, (x == 0)), dim=-1)
                    proto = torch.nn.functional.normalize(model.prototypes, dim=-1)
                    preds.append((emb @ proto.T).argmax(dim=-1).cpu().numpy())
            oof[test_idx] = np.concatenate(preds)
            print(f"  [{n_shot}-shot] fold {fold_id}: "
                  f"{len(support):,} support -> {len(test_idx):,} query cells")

        metrics = probe.score_predictions(y, oof)
        metrics["shots"] = n_shot
        rows.append(metrics)
        np.save(WORK / f"fewshot_{n_shot}shot_oof.npy", oof)
        print(f"[{n_shot}-shot] macro F1 {metrics['F1-Score']:.4f} | "
              f"balanced acc {metrics['Balanced Accuracy']:.4f}")

    out = protocol.RESULTS / "spatium_fewshot.csv"
    pd.DataFrame(rows).set_index("shots").to_csv(out)
    print(f"\nwrote few-shot sweep -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument(
        "--stage",
        choices=("all", "tokenize", "embed", "fewshot"),
        default="all",
    )
    args = parser.parse_args()

    token_path = WORK / "tokens.npy"
    if args.stage in ("all", "tokenize") or not token_path.exists():
        tokens = tokenize(build_anndata())
    else:
        tokens = np.load(token_path)

    if args.stage in ("all", "embed"):
        embed(tokens, device=args.device, batch_size=args.batch_size)
    if args.stage in ("all", "fewshot"):
        few_shot(tokens, device=args.device, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()

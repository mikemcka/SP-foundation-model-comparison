# Spatial-proteomics foundation model comparison

**Findings: [RESULTS.md](RESULTS.md).** This file is the protocol and the
how-to-run.

Four foundation models for spatial proteomics, scored on the same cells, the
same labels, the same spatial folds and the same probe:

| Model | Kind | Weights | Repo |
| --- | --- | --- | --- |
| **KRONOS2** | marker-aware vision ViT | gated HF repo `MahmoodLab/KRONOS2` | [CORAL](https://github.com/mahmoodlab/CORAL) (companion toolkit), [KRONOS](https://github.com/mahmoodlab/KRONOS) |
| **DeepCell Types** | language-informed vision | gated, `users.deepcell.org` | [deepcell-types](https://github.com/vanvalenlab/deepcell-types) |
| **Spatium** | protein-language transformer | vendored `Spatium/final.ckpt` | [Spatium](https://github.com/ploughhh/Spatium) |
| **VirTues** | tissue-scale, sequence-aware ViT | public HF repo `bunnelab/virtues` (`virtues-sp32`, CC BY-NC 4.0) | [virtues](https://github.com/bunnelab/virtues), [paper](https://www.nature.com/articles/s41586-026-10884-y) |

A **mean-marker** readout (per-cell average intensity per channel) is scored
alongside them. It is not a fifth competitor — it is the control that says
whether a foundation model is earning its keep on a well-designed antibody panel.

## The dataset

One classical Hodgkin lymphoma (cHL) region imaged by CODEX, released with
[MAPS](https://doi.org/10.1038/s41467-024-45999-1) and hosted on
[Zenodo](https://zenodo.org/records/10067010) (CC-BY-4.0, ~3.4 GB): one
single-channel TIFF per marker, a published cell segmentation mask, and
expert cell-type annotations.

The published mask is imported rather than re-segmented. The ground-truth labels
are defined on those exact cell ids, so re-running Cellpose would break the join
and quietly change what every model is scored on.

## The protocol

Adopted wholesale from CORAL's published cell-phenotyping benchmark (tutorial 5)
rather than reinvented, so the numbers here land on the same footing as the
published ones. It lives in [src/common/protocol.py](src/common/protocol.py) —
one file, so a difference in score cannot be a difference in protocol.

- **16 classes.** `Seg Artifact` dropped (those are segmentation failures, so
  scoring them measures the segmenter); `Cytotoxic CD8` merged into `CD8` (~380
  cells vs ~17k). `Other` is **kept** — it is the hardest class and dropping it
  would make the numbers incomparable.
- **18-marker panel**, fixed across all models. This is the only way the
  mean-marker control and the foundation models read the same channels.
- **Spatial folds.** Quarter the slide, hold out one quadrant at a time, with a
  64 px guard band either side of each midline. A random split would inflate
  every score: neighbouring cells share microenvironment and staining, and 64 px
  patches on centroids 20 px apart literally share pixels.
- **The probe.** Multinomial logistic regression on standardised features,
  `class_weight="balanced"`, training capped at 2000 cells per class (equal
  budget per encoder), `C` tuned by Optuna over `[1e-4, 1e2]` — 15 trials per
  fold, selected on validation macro F1, scored once on the held-out quadrant.
- **Metrics.** Macro F1 (headline, under this much imbalance), balanced
  accuracy, average precision, ROC AUC — mean ± std across folds, never a single
  fold.

### Two tables, two questions

The report separates these deliberately; conflating them is the easiest way to
draw a wrong conclusion from this repo.

1. **Linear probe on frozen features** — supervision-matched, so it ranks
   *representations*. Every encoder sees identical cells, folds, training budget
   and search space.
2. **Native usage mode** — DeepCell Types zero-shot, Spatium few-shot
   (10/20/100 cells per class). This is how each model is meant to be deployed,
   but the supervision differs between rows, so it ranks *deployments*, not
   representations. KRONOS2 and VirTues have no row here: for both, the
   documented cell-phenotyping mode *is* a linear probe on frozen features, so
   their native mode and the supervision-matched comparison are one measurement.

### Deviations from the original brief

- **No resample to 0.5 µm/px, except for VirTues.** The slide is kept at its
  native 0.37 mpp and each model is told that number, so each applies its own
  documented internal resampling. Pre-resampling would interpolate twice for
  DeepCell Types (which takes `mpp` as an argument and rescales itself) and
  would move KRONOS2 off the resolution its published benchmark was measured at.
  Identical pixels plus an identical truthful `mpp` *is* the harmonisation.
  **VirTues is the exception and has to be**: it takes no `mpp` argument and
  does no internal resampling, because its training corpus is stored at a fixed
  1.0 µm/px — its 8 px patch *is* an 8 µm patch. Handing it 0.37 mpp pixels
  would present every structure at 2.7× the scale it was pretrained on. It is
  therefore area-averaged down to 1.0 mpp, from the same ingested slide.
- **DeepCell Types is also probed, not only run zero-shot.** Its `cls_embedding`
  is exposed on the model output, so it can join the supervision-matched table.
  Without that, its only number would confound representation quality with
  vocabulary alignment.
- **Spatium is also probed, not only fine-tuned**, for the same reason.
- **VirTues' preprocessing is replicated, not imported.** Its shipped loader
  only reads datasets in spora's on-disk format, and converting this slide into
  that format would fork the mask and panel handling into a second
  implementation. `run_virtues.py` reproduces the four steps from VirTues' own
  `MultiplexDataset._preprocess` instead — clip at the per-image 99th
  percentile, log1p, 3×3 Gaussian blur, standardise — reading the one canonical
  slide and mask. Two judgement calls inside that:
  - The blur is in the *training* path but the phenotyping demo notebook
    standardises without it. The training path wins here (preprocess at
    inference as at pretraining); `--no-blur` runs the other reading.
  - spora's mean/std are corpus-level and its clipping quantile is per image.
    With one slide there is nothing else to compute either from, so both come
    from this slide.
- **VirTues cell tokens are pooled in one pass rather than via
  `compute_cell_tokens`.** Same crop grid (from its own `_get_uniform_crops`),
  same patch tokens, same pixel-overlap-weighted average — but accumulated into
  a running sum. The shipped helper holds every crop's tokens plus a few million
  single-token tensors in memory at once, which is tens of GB on a slide this
  size. It also skips crops that contain no scored cell, which changes no output
  because a crop only ever contributes to cells inside it.

### Marker harmonisation

Three separate name-resolution steps, all of them load-bearing.

**1. Acquisition names → CORAL's registry.** `coral ingest` is a two-pass
command: pass 1 reads channel names only and stops at a review gate for anything
it cannot resolve; pass 2 reads pixels. Four of this slide's 49 channels need a
human decision, applied in `MARKER_MAP_FIXES`
([src/prep_coral.py](src/prep_coral.py)):

| Raw name | Mapped to | Why |
| --- | --- | --- |
| `DAPI-01` | `DAPI` | per-cycle nuclear re-stain |
| `Cytokeritin` | `CYTOKERATIN` | vendor typo |
| `VISA` | `VISTA` | a.k.a. VSIR / B7-H5 |
| `Collagen 4` | `COLLAGENiv` | collagen **IV** — *not* the generic `COLLAGEN` CORAL suggests first |

**2 and 3. CORAL's names → each model's own vocabulary.** This is not cosmetic —
it decides what each model can see.

| Panel marker | KRONOS2 | DeepCell Types | Spatium | VirTues |
| --- | --- | --- | --- | --- |
| `dapi` | ✅ | ✅ as `dsDNA` | ❌ not a protein — outside a protein-language vocabulary | ❌ not a protein — no sequence to embed |
| `mct` (mast cell tryptase) | ✅ | ✅ as `Tryptase` | ❌ no tryptase entry | ✅ as `Q15661` (TPSAB1) |
| `cd30` | ✅ | ❌ **no CD30/TNFRSF8 entry** | ✅ | ✅ as `P28908` (TNFRSF8) |
| other 15 | ✅ | ✅ | ✅ | ✅ |
| **markers seen** | **18/18** | **17/18** | **16/18** | **17/18** |

KRONOS2 is marker-agnostic — it encodes whatever channels the patch carries — so
it sees 18/18. DeepCell Types sees 17/18 but is **missing the marker that defines
the Tumor (Hodgkin Reed-Sternberg) class in this dataset**. Spatium sees 16/18.
VirTues sees 17/18, losing DAPI. Aliases are declared explicitly in
[src/run_dct.py](src/run_dct.py); Spatium's own `protein_standard_mapping.tsv`
does the gene-symbol conversion (`cd20` → `MS4A1`, `cd11b` → `ITGAM`,
`cytokeratin` → `KRT`).

VirTues resolves markers by **protein sequence, not by name**: each channel is
embedded with ESM-2 (`esm2_t30_150M_UR50D`, the instance the released weights
were trained with) from the canonical UniProt sequence of its target, and that
vector is the only handle the model has on the channel's identity. So the panel
has to be mapped onto accessions before anything runs
([src/common/virtues_markers.yaml](src/common/virtues_markers.yaml)), a channel
whose target is not a protein has nothing to embed, and there is no name-matching
fallback. Two of the 17 need a decision rather than a lookup, and both copy
spora's own channel table — the accessions the released weights were trained
under — rather than being re-derived:

| Panel marker | Accession | Why it is not obvious |
| --- | --- | --- |
| `cd15` | `P22083` (FUT4) | CD15 is the Lewis-X carbohydrate epitope, not a protein; spora maps it to the fucosyltransferase that synthesises it |
| `cytokeratin` | `P02533` (KRT14) | pan-CK antibodies bind many keratins; spora resolves its pan-CK channel to KRT14 |

DeepCell Types' 51-class vocabulary is mapped onto the 16 cHL classes in
[src/common/dct_class_map.yaml](src/common/dct_class_map.yaml), by one uniform
rule: *map to the label a cHL annotator using this 18-marker panel would have
assigned*. Two classes cannot be mapped because its vocabulary is **coarser**
than the benchmark's — `Macrophage` (the benchmark splits M1/M2) and generic
`Tcell` — so `evaluate.py` also reports a collapsed label space with M1+M2
merged, to separate a vocabulary artifact from a perception failure.

## Running it

```bash
source env/secrets.env                 # HF_TOKEN + DEEPCELL_ACCESS_TOKEN, gitignored
GPU_JOBID=<your gpuq job> ./run_all.sh # or leave unset to sbatch each GPU stage
```

Individual stages: `./run_all.sh prep | kronos2 | dct | spatium | virtues |
evaluate`. Stages hand off through files under `results/`, never through a live
process, so any one can be re-run alone.

VirTues splits into two: `--stage markers` (CPU, needs internet — UniProt plus
the ESM-2 download) and `--stage tokens` (GPU, no internet beyond the
checkpoint). `run_all.sh` runs the first locally and submits the second, because
the GPU nodes here have no outbound network.

### Environments

Four, because the models pin incompatible torch stacks. Built by
`env/build_*.sh` into `/vast/scratch/users/mckay.m/condaenvs`.

| Env | For | Stack |
| --- | --- | --- |
| `sp-coral` | prep, KRONOS2, evaluation | torch 2.6.0+cu124, transformers 4.56.0, timm 1.0.19 — pinned exactly; KRONOS2 embeddings are only bit-exact to the published gold standard on this stack, at `--batch-size 16` |
| `sp-dct` | DeepCell Types | torch 2.13+cu130 |
| `sp-spatium` | Spatium | torch 2.6.0+cu124, pytorch-lightning, zarr <3 |
| `sp-virtues` | VirTues | python 3.12, torch+cu126, flash-attn, fair-esm — VirTues' own `setup.sh` choices |

### Gotchas worth knowing

- **Both checkpoints are gated.** `MahmoodLab/KRONOS2` is manual-approval on
  Hugging Face; DeepCell Types needs a free `users.deepcell.org` token. Tokens
  live in `env/secrets.env` (gitignored, `chmod 600`).
- **`~/.deepcell` is a symlink to scratch.** `deepcell_auth` hardcodes the cache
  to `$HOME/.deepcell` and the home quota is too small for it.
- **Spatium ships no packaging.** No `pyproject.toml`, no install docs, and its
  modules import each other by bare name, so `Spatium/src` goes on `sys.path` as
  a root. Its `dataloader.py` also imports NVIDIA Merlin at module scope while
  only its unused parquet path needs it —
  [src/spatium_shim.py](src/spatium_shim.py) stubs it so the imports resolve and
  any real use raises loudly.
- **Spatium needs per-marker z-scoring.** Only the rank *order* of markers
  survives tokenisation, so on raw intensities the ranking is dominated by which
  antibodies are globally bright — nearly identical for every cell.
- **flash-attn is not optional for VirTues, and it is the slow build.** Its
  attention blocks import `flash_attn.flash_attn_interface` at module scope, so
  the package will not import without it, and it is CUDA-only — the token stage
  cannot fall back to CPU. `pip install flash-attn --no-build-isolation`
  compiles kernels; budget 30-60 min unless a matching wheel exists.
- **`virtues-sp32` is CC BY-NC 4.0** — academic use only. `virtues-sp31` (MIT,
  trained on 31 of the 32 datasets) is the commercial-use alternative; swap
  `HF_WEIGHTS` in [src/run_virtues.py](src/run_virtues.py).
- **VirTues' marker embedding directory is read wholesale.** Its loader takes
  every `.pt` in the directory and orders them by filename, so a stale embedding
  left over from a different panel silently shifts every marker index by one.
  `run_virtues.py` asserts the set on disk matches the panel exactly.

## Layout

```
src/
  common/protocol.py         the shared contract: classes, panel, folds, probe settings,
                             and the one definition of which slide planes are the panel
  common/probe.py            scoring harness (wraps CORAL's own probe helpers)
  common/dct_class_map.yaml  DeepCell Types vocabulary -> cHL classes
  common/virtues_markers.yaml panel -> UniProt accessions for VirTues
  prep_coral.py              ingest -> mask/label import -> patches -> canonical cell table
  run_kronos2.py             KRONOS2 per-cell CLS embeddings
  run_dct.py                 DeepCell Types zero-shot + frozen CLS embeddings
  run_spatium.py             rank tokens -> frozen embeddings + few-shot fine-tune
  spatium_shim.py            makes Spatium's bare source tree importable
  run_virtues.py             ESM-2 marker embeddings -> whole-slide sweep -> cell tokens
  evaluate.py                scores everything, writes tables and figures
results/
  cells.parquet             THE canonical cell list — every model aligns to it by cell_id
  features/*.npz            one per encoder: cell_id + features
  predictions/              native-mode calls
  virtues/                  ESM-2 marker embeddings, FASTAs, VirTues checkpoint
  figures/                  probe summary, per-class F1, confusion, spatial errors
RESULTS.md                  the findings
```

`results/cells.parquet` is the load-bearing artifact. Every model writes features
or predictions keyed by `cell_id` and the evaluator reindexes onto that table —
no script may assume its own natural row order matches another's.

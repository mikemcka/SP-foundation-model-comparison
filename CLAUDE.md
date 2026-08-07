# Spatial-proteomics foundation model comparison

**Findings: [RESULTS.md](RESULTS.md).** This file is the protocol and the
how-to-run.

Three foundation models for spatial proteomics, scored on the same cells, the
same labels, the same spatial folds and the same probe:

| Model | Kind | Weights | Repo |
| --- | --- | --- | --- |
| **KRONOS2** | marker-aware vision ViT | gated HF repo `MahmoodLab/KRONOS2` | [CORAL](https://github.com/mahmoodlab/CORAL) (companion toolkit), [KRONOS](https://github.com/mahmoodlab/KRONOS) |
| **DeepCell Types** | language-informed vision | gated, `users.deepcell.org` | [deepcell-types](https://github.com/vanvalenlab/deepcell-types) |
| **Spatium** | protein-language transformer | vendored `Spatium/final.ckpt` | [Spatium](https://github.com/ploughhh/Spatium) |

A **mean-marker** readout (per-cell average intensity per channel) is scored
alongside them. It is not a fourth competitor — it is the control that says
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
   representations.

### Deviations from the original brief

- **No resample to 0.5 µm/px.** The slide is kept at its native 0.37 mpp and
  each model is told that number, so each applies its own documented internal
  resampling. Pre-resampling would interpolate twice for DeepCell Types (which
  takes `mpp` as an argument and rescales itself) and would move KRONOS2 off the
  resolution its published benchmark was measured at. Identical pixels plus an
  identical truthful `mpp` *is* the harmonisation.
- **DeepCell Types is also probed, not only run zero-shot.** Its `cls_embedding`
  is exposed on the model output, so it can join the supervision-matched table.
  Without that, its only number would confound representation quality with
  vocabulary alignment.
- **Spatium is also probed, not only fine-tuned**, for the same reason.

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

| Panel marker | KRONOS2 | DeepCell Types | Spatium |
| --- | --- | --- | --- |
| `dapi` | ✅ | ✅ as `dsDNA` | ❌ not a protein — outside a protein-language vocabulary |
| `mct` (mast cell tryptase) | ✅ | ✅ as `Tryptase` | ❌ no tryptase entry |
| `cd30` | ✅ | ❌ **no CD30/TNFRSF8 entry** | ✅ |
| other 15 | ✅ | ✅ | ✅ |

KRONOS2 is marker-agnostic — it encodes whatever channels the patch carries — so
it sees 18/18. DeepCell Types sees 17/18 but is **missing the marker that defines
the Tumor (Hodgkin Reed-Sternberg) class in this dataset**. Spatium sees 16/18.
Aliases are declared explicitly in [src/run_dct.py](src/run_dct.py); Spatium's own
`protein_standard_mapping.tsv` does the gene-symbol conversion (`cd20` → `MS4A1`,
`cd11b` → `ITGAM`, `cytokeratin` → `KRT`).

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

Individual stages: `./run_all.sh prep | kronos2 | dct | spatium | evaluate`.
Stages hand off through files under `results/`, never through a live process, so
any one can be re-run alone.

### Environments

Three, because the models pin incompatible torch stacks. Built by
`env/build_*.sh` into `/vast/scratch/users/mckay.m/condaenvs`.

| Env | For | Stack |
| --- | --- | --- |
| `sp-coral` | prep, KRONOS2, evaluation | torch 2.6.0+cu124, transformers 4.56.0, timm 1.0.19 — pinned exactly; KRONOS2 embeddings are only bit-exact to the published gold standard on this stack, at `--batch-size 16` |
| `sp-dct` | DeepCell Types | torch 2.13+cu130 |
| `sp-spatium` | Spatium | torch 2.6.0+cu124, pytorch-lightning, zarr <3 |

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

## Layout

```
src/
  common/protocol.py        the shared contract: classes, panel, folds, probe settings
  common/probe.py           scoring harness (wraps CORAL's own probe helpers)
  common/dct_class_map.yaml DeepCell Types vocabulary -> cHL classes
  prep_coral.py             ingest -> mask/label import -> patches -> canonical cell table
  run_kronos2.py            KRONOS2 per-cell CLS embeddings
  run_dct.py                DeepCell Types zero-shot + frozen CLS embeddings
  run_spatium.py            rank tokens -> frozen embeddings + few-shot fine-tune
  spatium_shim.py           makes Spatium's bare source tree importable
  evaluate.py               scores everything, writes tables and figures
results/
  cells.parquet             THE canonical cell list — every model aligns to it by cell_id
  features/*.npz            one per encoder: cell_id + features
  predictions/              native-mode calls
  figures/                  probe summary, per-class F1, confusion, spatial errors
RESULTS.md                  the findings
```

`results/cells.parquet` is the load-bearing artifact. Every model writes features
or predictions keyed by `cell_id` and the evaluator reindexes onto that table —
no script may assume its own natural row order matches another's.

# The four models, and why they disagree

Background notes on what each foundation model actually encodes. The four take
genuinely different views of the same cell, which is what makes comparing them
interesting — and what makes a single ranking misleading on its own. For the
benchmark protocol and how to run it, see [CLAUDE.md](CLAUDE.md); for the
numbers, [RESULTS.md](RESULTS.md).

## 1. KRONOS — vision-based morphological embeddings

Image-centric: trained on over 47 million multiplexed image patches.

- **Input:** multi-channel image patches (raw pixels).
- **Features:** morphological and spatial structure from the pixels themselves.
- **Scale:** because it works on images, it embeds at any scale — single cells,
  microenvironments, whole tissue regions.
- **Key innovation:** a shared convolutional filter across markers plus
  sinusoidal marker encodings, which keeps it *agnostic to which markers* a
  panel happens to carry.

## 2. DeepCell Types — language-informed vision embeddings

Combines visual information with semantic biological knowledge.

- **Input:** 64×64 cell patches augmented with segmentation masks.
- **Features:** a language encoder (an LLM) produces a semantic embedding for
  each protein marker, from its biological function and alternative names.
- **Integration:** a channel-wise transformer fuses the visual and linguistic
  features, so the embedding represents both what the cell looks like and what
  the markers it expresses *mean*.
- **Key innovation:** LLM-generated semantic embeddings let the model
  "understand" marker identity rather than treat channels as anonymous.

## 3. Spatium — protein-language expression embeddings

A protein language model: it treats expression profiles the way an LLM treats
text. Trained on the expression matrices of over 51 million cells.

- **Input:** per-cell protein abundance — no pixels at all.
- **Features:** rank-based tokenisation, ordering proteins by relative abundance
  within each cell.
- **Focus:** co-expression hierarchies and combinatorial marker patterns —
  molecular phenotype rather than visual morphology.
- **Key innovation:** ranked protein sequences as a "language" of cell state.

## 4. VirTues — tissue-scale, sequence-aware embeddings

*[Virtual Tissues](https://www.nature.com/articles/s41586-026-10884-y), Nature
2026.* Learns across scales — proteins, cells, niches, whole tissues — from one
pretrained backbone, trained on 32 spatial-proteomics datasets spanning
different platforms and panels.

- **Input:** whole 128×128 tissue tiles, not cell crops. A cell has no
  representation of its own until one is pooled out of the tile.
- **Features:** alternating spatial and *marker* attention produces one summary
  token per 8×8 patch; a cell token is the pixel-weighted average of the patch
  tokens its segmentation mask overlaps. Every cell token therefore carries its
  neighbourhood by construction — the only model here for which that is true.
- **Marker identity:** each channel is embedded by **ESM-2 from the amino-acid
  sequence of its target protein**. Not a name lookup and not a learned
  per-marker code: an unseen antibody is handled by embedding its target's
  sequence, which is what lets one backbone read panels it was never trained on.
- **Key innovation:** marker awareness grounded in protein sequence, plus a
  single backbone that scales from a patch to a whole tissue.

## Summary

| Model | Primary input | Key innovation |
| --- | --- | --- |
| **KRONOS** | image patches (pixels) | shared convolutional weights + sinusoidal marker encodings |
| **DeepCell Types** | image + language (semantic) | LLM-generated semantic embeddings to "understand" marker identity |
| **Spatium** | protein abundance (ranked) | ranked protein sequences as a "language" of cell state |
| **VirTues** | whole tissue tiles (pixels) | ESM-2 sequence embeddings as marker identity; one backbone from patch to tissue |

Different as their approaches are, all four provide a transferable latent space
for classifying cell types, identifying spatial niches, and detecting patterns
across datasets and imaging platforms.

Two axes separate them, and both matter for reading the results. **What they
read:** pixels (KRONOS, DeepCell Types, VirTues) against per-cell expression
(Spatium). **What they read it over:** a single cell (KRONOS, DeepCell Types,
Spatium) against a tissue region (VirTues) — so VirTues is the only one that can
use a cell's surroundings, and the only one whose score is not purely a property
of the cell itself.

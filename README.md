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

## 4. VirTues — whole-tissue attention with protein-sequence-keyed channels

Tissue-centric: trained on a multi-technology corpus of whole-slide multiplex
images spanning IMC, CODEX, MIBI and Orion.

- **Input:** whole-tissue image crops (128×128 px at 1.0 µm/px), not individual
  cells — a cell's representation is pooled afterward from the patch tokens its
  segmentation mask overlaps, not read out directly.
- **Features:** alternating spatial and marker attention across the crop, so
  every patch token is shaped by both its neighbourhood and every other channel
  measured at that position.
- **Scale:** tissue context by construction — there is no isolated single-cell
  view, only cell-, niche- and tissue-level poolings of the same underlying
  patch tokens.
- **Key innovation:** channels are identified to the model by an ESM-2
  embedding of their target protein's amino-acid sequence — the same
  protein-language idea Spatium uses for expression, but fused into spatial
  image tokens rather than replacing them. Like KRONOS, it accepts any
  combination of markers a panel happens to carry; unlike Spatium, it does that
  without giving up pixels.

## Summary

| Model | Primary input | Key innovation |
| --- | --- | --- |
| **KRONOS** | image patches (pixels) | shared convolutional weights + sinusoidal marker encodings |
| **DeepCell Types** | image + language (semantic) | LLM-generated semantic embeddings to "understand" marker identity |
| **Spatium** | protein abundance (ranked) | ranked protein sequences as a "language" of cell state |
| **VirTues** | whole-tissue image crops (pixels) | ESM-2 protein-sequence embeddings fused into spatial attention tokens |

Different as their approaches are, all four provide a transferable latent space
for classifying cell types, identifying spatial niches, and detecting patterns
across datasets and imaging platforms.

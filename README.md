# The three models, and why they disagree

Background notes on what each foundation model actually encodes. The three take
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

## Summary

| Model | Primary input | Key innovation |
| --- | --- | --- |
| **KRONOS** | image patches (pixels) | shared convolutional weights + sinusoidal marker encodings |
| **DeepCell Types** | image + language (semantic) | LLM-generated semantic embeddings to "understand" marker identity |
| **Spatium** | protein abundance (ranked) | ranked protein sequences as a "language" of cell state |

Different as their approaches are, all three provide a transferable latent space
for classifying cell types, identifying spatial niches, and detecting patterns
across datasets and imaging platforms.

#!/usr/bin/env bash
# Build the VirTues environment.
#
# Python 3.12 and the cu126 torch index are VirTues' own choices (its setup.sh);
# they are not shared with the other three envs, which is the whole reason each
# model gets its own.
#
# flash-attn is the long pole and is not optional: VirTues' attention blocks
# import `flash_attn.flash_attn_interface` at module scope, so the package fails
# to import without it. --no-build-isolation is required (the build needs the
# already-installed torch) and it compiles CUDA kernels, so budget ~30-60 min on
# a build node with nvcc. If a prebuilt wheel exists for this python/torch/CUDA
# combination, install that instead — it is the same package.
#
# spora-io is deliberately NOT installed. It only reads datasets stored in
# spora's on-disk format; this slide is a CORAL store, and converting it would
# fork the mask and panel handling into a second implementation. run_virtues.py
# reproduces the four preprocessing steps from VirTues' own MultiplexDataset
# instead.
#
# Base CORAL (no extras, so no torch pin) so this env can read the ingested
# slide — same reason as sp-dct.
set -euo pipefail

REPO=/vast/scratch/users/mckay.m/SP-foundation-model-comparison
ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
ENV_NAME=sp-virtues

source /etc/profile.d/modules.sh
module load miniconda3

conda create -y -p "$ENV_ROOT/$ENV_NAME" python=3.12
# shellcheck disable=SC1091
source activate "$ENV_ROOT/$ENV_NAME"

python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install \
    einops omegaconf loguru safetensors huggingface_hub \
    "fair-esm" biopython requests \
    numpy pandas pyarrow scikit-learn matplotlib tifffile

pip install flash-attn --no-build-isolation

# VirTues itself, editable from the local clone.
pip install -e "$REPO/virtues"

pip install -e "$REPO/CORAL"

python - <<'PY'
import torch
from virtues.modules.multiplex_virtues import MultiplexVirtues
print("torch    ", torch.__version__, "cuda build", torch.version.cuda)
print("virtues  ", MultiplexVirtues.__module__)
import esm, coral
print("fair-esm ", esm.__file__)
print("coral    ", coral.__file__)
PY
echo "BUILD_OK sp-virtues"

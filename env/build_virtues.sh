#!/usr/bin/env bash
# Build the VirTues environment.
#
# The pins are the paper's own stack (Methods, "Computing hardware and
# software": Python 3.12.9, PyTorch 2.5.1 + CUDA 12.1, Flash Attention-2
# v2.7.4), which is what the released weights were run under. VirTues' own
# pyproject declares no runtime dependencies at all and its
# version_requirements.txt is a `pip list` dump of the authors' NVIDIA dev
# container, so there is nothing authoritative to resolve against — the pins
# below are lifted from VirTues-Nextflow's module environments, which already
# did this work.
#
# ONE env, not two, unlike the Nextflow pipeline. It splits marker embedding
# (CPU torch + fair-esm) from encoding (CUDA torch + flash-attn) because its GPU
# nodes have no outbound network. Here both stages run from the same script and
# fair-esm is indifferent to which torch it finds, so a second env would only
# add a second thing to keep in sync.
#
# flash-attn is NOT optional: VirTues' attention blocks import
# flash_attn.flash_attn_interface at module scope, so `import
# virtues.modules.multiplex_virtues` fails outright without it. There is no CPU
# fallback. It is installed from a prebuilt wheel rather than by name because
# building it needs `pip install --no-build-isolation` (the build imports the
# already-installed torch) and takes 30-60 minutes of CUDA compilation. The
# wheel must match the python/torch/CUDA triple above.
#
# Consequence for scheduling: flash-attn 2.x needs compute capability >= 8.0,
# so this env runs on the A100/A30/A10 nodes and NOT on gpuq's P100s, where it
# fails at import. run_all.sh requests an Ampere card explicitly.
#
# VirTues itself is not pip-installed. Its pyproject declares only the
# top-level package (`include = ["virtues"]`) and it ships no __init__.py for
# virtues.utils, so a non-editable install does not carry what we import; the
# authors' own instruction is `pip install -e .`. run_virtues.py puts the clone
# on sys.path instead, which does the same thing and pins the exact source.
set -euo pipefail

ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
ENV_NAME=sp-virtues

FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

source /etc/profile.d/modules.sh
module load miniconda3

conda create -y -p "$ENV_ROOT/$ENV_NAME" python=3.12
# shellcheck disable=SC1091
source activate "$ENV_ROOT/$ENV_NAME"

python -m pip install --upgrade pip

pip install \
    "torch==2.5.1" "torchvision==0.20.1" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# einops/omegaconf/safetensors/loguru are VirTues' own imports; tifffile and
# pillow are the wrapper's (it reads the panel TIFF and the label mask).
pip install \
    einops omegaconf safetensors loguru \
    "tifffile>=2023.0" "pillow>=10" \
    numpy pandas pyarrow scipy scikit-image tqdm \
    huggingface_hub

# esm2_t30_150M_UR50D is the instance the released weights were trained with:
# its 640-d output is what the model's prior_embedding_encoder is shaped for, so
# fair-esm's version is a compatibility pin, not a preference.
pip install "fair-esm==2.0.0" biopython requests

pip install "$FLASH_ATTN_WHEEL"

python - <<'PY'
import torch, flash_attn, esm, tifffile, omegaconf, safetensors
print("torch      ", torch.__version__, "cuda build", torch.version.cuda)
print("flash-attn ", flash_attn.__version__)
print("fair-esm   ", esm.__version__ if hasattr(esm, "__version__") else "2.0.0")
print("tifffile   ", tifffile.__version__)
PY
echo "BUILD_OK sp-virtues"

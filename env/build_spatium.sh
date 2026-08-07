#!/usr/bin/env bash
# Build the Spatium environment.
#
# Spatium ships as a bare src/ tree with no packaging metadata and no
# installation docs, so the dependency set below is read off its imports:
#   model.py / fine_tune.py -> torch, pytorch_lightning, transformers, torchmetrics
#   tokenizer.py            -> scanpy, anndata, numba, pyarrow, scipy
#   dataloader.py           -> zarr, sklearn  (+ merlin, see note)
#
# NOTE on merlin: dataloader.py imports NVIDIA Merlin at module scope, but only
# its parquet path uses it — FewShotTokenDataset / MerlinFewShotDataModule (the
# few-shot path we need) are pure zarr + torch. Merlin is a heavy, CUDA-pinned
# stack that does not resolve against torch 2.6, so we stub it at import time
# (see src/spatium_shim.py) rather than install it.
#
# zarr is pinned <3: Spatium calls the v2 `Group.create_dataset` / `Array.append`
# API, which zarr 3 removed.
#
# optuna + ipython are not Spatium's: the few-shot stage imports the shared
# scoring harness (src/common/probe.py), which wraps CORAL's tutorial probe
# helpers, and those pull optuna (the C search) and IPython (a re-exported
# notebook display helper).
set -euo pipefail

ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
ENV_NAME=sp-spatium

source /etc/profile.d/modules.sh
module load miniconda3

conda create -y -p "$ENV_ROOT/$ENV_NAME" python=3.11
# shellcheck disable=SC1091
source activate "$ENV_ROOT/$ENV_NAME"

python -m pip install --upgrade pip

pip install \
    "torch==2.6.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124

pip install \
    "pytorch-lightning>=2.2,<3" \
    "torchmetrics>=1.4" \
    "transformers>=4.40" \
    "zarr>=2.18,<3" \
    "numcodecs<0.16" \
    scanpy anndata numba pyarrow scipy scikit-learn pandas matplotlib tqdm \
    optuna ipython

python - <<'PY'
import torch, pytorch_lightning as pl, zarr, scanpy, numba
print("torch    ", torch.__version__, "cuda build", torch.version.cuda)
print("lightning", pl.__version__)
print("zarr     ", zarr.__version__)
print("scanpy   ", scanpy.__version__)
PY
echo "BUILD_OK sp-spatium"

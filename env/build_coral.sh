#!/usr/bin/env bash
# Build the CORAL / KRONOS2 environment.
#
# KRONOS2's numeric stack is pinned exactly by CORAL's `kronos2` extra
# (torch 2.6.0 / xformers 0.0.29.post3 / transformers 4.56 / timm 1.0.19):
# embeddings are only bit-exact to the published gold standard on that stack.
# torch 2.6.0 resolves to cu124, and xformers needs the matching cu124 wheel
# from the PyTorch index, hence --extra-index-url.
set -euo pipefail

REPO=/vast/scratch/users/mckay.m/SP-foundation-model-comparison
ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
ENV_NAME=sp-coral

source /etc/profile.d/modules.sh
module load miniconda3

conda create -y -p "$ENV_ROOT/$ENV_NAME" python=3.11
# shellcheck disable=SC1091
source activate "$ENV_ROOT/$ENV_NAME"

python -m pip install --upgrade pip

# CORAL + the KRONOS2 extra, from the local clone.
pip install -e "$REPO/CORAL[kronos2]" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Tutorial-side analysis deps (linear probe, hyperparameter search, plots).
# ipython is not used directly: CORAL's tutorials/utils/__init__.py re-exports a
# notebook display helper, so importing its probe protocol pulls IPython in.
pip install scikit-learn optuna seaborn anndata ipython

python - <<'PY'
import torch, transformers, timm
print("torch      ", torch.__version__, "cuda build", torch.version.cuda)
print("transformers", transformers.__version__)
print("timm       ", timm.__version__)
import coral
print("coral      ", coral.__file__)
PY
echo "BUILD_OK sp-coral"

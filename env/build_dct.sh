#!/usr/bin/env bash
# Build the DeepCell Types environment.
#
# Inference needs only the checkpoint plus the packaged vocab.json, so the base
# install is enough — no [train] extra, no TissueNet zarr archive. The
# checkpoint itself is gated: `deepcell_auth` fetches it from
# users.deepcell.org and requires DEEPCELL_ACCESS_TOKEN (env/secrets.env).
set -euo pipefail

REPO=/vast/scratch/users/mckay.m/SP-foundation-model-comparison
ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
ENV_NAME=sp-dct

source /etc/profile.d/modules.sh
module load miniconda3

conda create -y -p "$ENV_ROOT/$ENV_NAME" python=3.11
# shellcheck disable=SC1091
source activate "$ENV_ROOT/$ENV_NAME"

python -m pip install --upgrade pip

# Editable install of the local clone; deepcell_auth comes from git per the
# project's own dependency pin.
pip install -e "$REPO/deepcell-types"

# Shared evaluation deps (the comparison harness scores every model the same way).
pip install scikit-learn pandas matplotlib tifffile

# Base CORAL (no extras, so no torch pin) so this env can read the ingested
# slide. DeepCell Types must see exactly the planes KRONOS2 saw; re-deriving the
# 18-marker stack from the raw per-channel TIFFs here would fork the marker
# mapping into a second implementation, which is precisely how a panel silently
# drifts between models.
pip install -e "$REPO/CORAL"

python - <<'PY'
import torch
from deepcell_types.utils import list_model_versions, list_supported_cell_types
print("torch        ", torch.__version__, "cuda build", torch.version.cuda)
print("model versions", list_model_versions())
print("n cell types ", len(list_supported_cell_types()))
PY
echo "BUILD_OK sp-dct"

#!/usr/bin/env bash
# End-to-end driver: raw data -> canonical cells -> three encoders -> report.
#
# Each stage runs in its own conda env because the three models pin
# incompatible torch stacks (KRONOS2 needs exactly torch 2.6.0+cu124; DeepCell
# Types resolves to cu13; Spatium wants pytorch-lightning on top of 2.6). Stages
# hand off through files under results/, never through a live process, so any
# one of them can be re-run alone.
#
# GPU stages run on Slurm. Set GPU_JOBID to attach to an existing interactive
# allocation (`srun --jobid=...`); leave it unset to submit each stage with
# sbatch --wait instead.
#
#   source env/secrets.env
#   GPU_JOBID=29122168 ./run_all.sh
#
# Individual stages: ./run_all.sh prep | kronos2 | dct | spatium | evaluate
set -euo pipefail

REPO=/vast/scratch/users/mckay.m/SP-foundation-model-comparison
ENV_ROOT=/vast/scratch/users/mckay.m/condaenvs
LOGS="$REPO/results/logs"
mkdir -p "$LOGS"

cd "$REPO"

# Only the gated downloads need credentials: MahmoodLab/KRONOS2 is
# manual-approval on Hugging Face and DeepCell Types wants a users.deepcell.org
# token. VirTues' weights are public (`bunnelab/virtues`), and the evaluator only
# reads files, so demanding tokens for those stages would be a wrong error.
case "${1:-all}" in
    all|prep|gpu|kronos2|dct)
        if [[ -z "${HF_TOKEN:-}" || -z "${DEEPCELL_ACCESS_TOKEN:-}" ]]; then
            echo "error: run 'source env/secrets.env' first (KRONOS2 and DeepCell" \
                 "Types are both gated downloads)." >&2
            exit 1
        fi
        ;;
esac

# Call an env's interpreter by absolute path rather than activating it.
# `sbatch --wrap` runs its script under /bin/sh, where `module load miniconda3`
# warns "Shell sh not recognised, conda will not be initialised" and leaves
# `source activate` resolving against PATH — which works or doesn't depending on
# what else the node has loaded. A conda env's python already knows its own
# site-packages, and these torch wheels resolve their CUDA libs through RPATH,
# so activation buys nothing here and costs reliability.
py() { echo "$ENV_ROOT/$1/bin/python"; }

# Run a python script inside one of the three envs.
in_env() {
    local env_name=$1; shift
    "$(py "$env_name")" "$@"
}

# Same, but on a GPU: either inside an existing allocation or via a fresh job.
on_gpu() {
    local env_name=$1; shift
    if [[ -n "${GPU_JOBID:-}" ]]; then
        srun --jobid="$GPU_JOBID" --export=ALL --chdir="$REPO" bash -c "$*"
    else
        # 8 cores, not 4: every stage streams 152k random 64px patch reads off
        # /vast, and the loaders are what keep the GPU fed.
        sbatch --wait --partition=gpuq --gres=gpu:1 --cpus-per-task=8 \
            --mem=120G --time=8:00:00 --job-name="sp-$env_name" \
            --output="$LOGS/%x-%j.out" --export=ALL --chdir="$REPO" \
            --wrap="$*"
    fi
}

# VirTues needs its own launcher because flash-attn is not optional for it:
# VirTues' attention blocks import flash_attn at module scope, and flash-attn 2.x
# needs compute capability >= 8.0. gpuq holds A30s and A100s (both 8.0) but also
# P100s (6.0), where the import fails outright — so the card is requested by name
# rather than left to the scheduler. A100 first because the resampled stack plus
# the token accumulators want the headroom.
on_ampere() {
    local env_name=$1; shift
    if [[ -n "${GPU_JOBID:-}" ]]; then
        srun --jobid="$GPU_JOBID" --export=ALL --chdir="$REPO" bash -c "$*"
    else
        sbatch --wait --partition=gpuq --gres=gpu:A100:1 --cpus-per-task=8 \
            --mem=256G --time=8:00:00 --job-name="sp-$env_name" \
            --output="$LOGS/%x-%j.out" --export=ALL --chdir="$REPO" \
            --wrap="$*"
    fi
}

# The probe is the one CPU-heavy stage: 15 Optuna trials x 4 folds x 4 encoders
# of multinomial logistic regression on up to 32k x 768. Give it real cores —
# the solver is BLAS-bound, so it scales with threads.
on_cpu() {
    local env_name=$1; shift
    if [[ -n "${NO_SLURM:-}" ]]; then
        bash -c "$*"
    else
        sbatch --wait --partition=regular --cpus-per-task=16 --mem=64G \
            --time=8:00:00 --job-name="sp-evaluate" --output="$LOGS/%x-%j.out" \
            --export=ALL --chdir="$REPO" --wrap="$*"
    fi
}

stage_prep()     { in_env sp-coral   src/prep_coral.py              2>&1 | tee "$LOGS/prep.log"; }
stage_kronos2()  { on_gpu  sp-coral  "$(py sp-coral) src/run_kronos2.py"  2>&1 | tee "$LOGS/kronos2.log"; }
stage_dct()      { on_gpu  sp-dct    "$(py sp-dct) src/run_dct.py"      2>&1 | tee "$LOGS/dct.log"; }
stage_spatium()  { on_gpu  sp-spatium "$(py sp-spatium) src/run_spatium.py" 2>&1 | tee "$LOGS/spatium.log"; }
# VirTues is three stages on three kinds of machine, not one: the panel export
# needs CORAL (sp-coral) to read the ingested zarr, the marker stage needs
# outbound network for UniProt and the ESM-2 weights, and only the encode needs a
# GPU. Splitting them means a GPU job is never queued behind a download.
stage_virtues_panel()   { in_env sp-coral   src/run_virtues.py --stage panel   2>&1 | tee "$LOGS/virtues-panel.log"; }
stage_virtues_markers() { in_env sp-virtues src/run_virtues.py --stage markers 2>&1 | tee "$LOGS/virtues-markers.log"; }
stage_virtues_embed()   { on_ampere sp-virtues "$(py sp-virtues) src/run_virtues.py --stage embed" 2>&1 | tee "$LOGS/virtues-embed.log"; }
stage_virtues()         { stage_virtues_panel; stage_virtues_markers; stage_virtues_embed; }

stage_evaluate() { on_cpu  sp-coral  "$(py sp-coral) src/evaluate.py"      2>&1 | tee "$LOGS/evaluate.log"; }

# The three GPU stages are independent — each reads prep's outputs and writes
# its own feature file — so fire them as separate jobs rather than queueing them
# behind each other. KRONOS2 is the long pole (a ViT-B forward per cell), so it
# gets the longer wall-clock request.
stage_gpu_parallel() {
    local common=(--partition=gpuq --gres=gpu:1 --cpus-per-task=8 --mem=120G
                  --output="$LOGS/%x-%j.out" --export=ALL)
    submit() {
        local name=$1 env_name=$2 time_limit=$3 script=$4
        sbatch "${common[@]}" --job-name="$name" --time="$time_limit" \
            --chdir="$REPO" --wrap="$(py "$env_name") $script"
    }
    submit sp-kronos2 sp-coral   8:00:00 src/run_kronos2.py
    submit sp-dct     sp-dct     4:00:00 src/run_dct.py
    submit sp-spatium sp-spatium 4:00:00 src/run_spatium.py
    squeue -u "$USER" -o "%.10i %.12j %.2t %.11M %R"
}

case "${1:-all}" in
    prep)     stage_prep ;;
    gpu)      stage_gpu_parallel ;;
    kronos2)  stage_kronos2 ;;
    dct)      stage_dct ;;
    spatium)  stage_spatium ;;
    virtues)  stage_virtues ;;
    virtues-panel)   stage_virtues_panel ;;
    virtues-markers) stage_virtues_markers ;;
    virtues-embed)   stage_virtues_embed ;;
    evaluate) stage_evaluate ;;
    all)
        stage_prep
        stage_kronos2
        stage_dct
        stage_spatium
        stage_virtues
        stage_evaluate
        ;;
    *)
        echo "usage: $0 [all|prep|gpu|kronos2|dct|spatium|virtues|evaluate]" >&2
        echo "  gpu     = submit the three patch-model stages as parallel Slurm jobs" >&2
        echo "  virtues = panel export, then markers (needs network), then GPU encode" >&2
        echo "            sub-stages: virtues-panel | virtues-markers | virtues-embed" >&2
        exit 2
        ;;
esac

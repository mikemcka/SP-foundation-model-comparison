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

if [[ -z "${HF_TOKEN:-}" || -z "${DEEPCELL_ACCESS_TOKEN:-}" ]]; then
    echo "error: run 'source env/secrets.env' first (both model downloads are gated)." >&2
    exit 1
fi

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
    evaluate) stage_evaluate ;;
    all)
        stage_prep
        stage_kronos2
        stage_dct
        stage_spatium
        stage_evaluate
        ;;
    *)
        echo "usage: $0 [all|prep|gpu|kronos2|dct|spatium|evaluate]" >&2
        echo "  gpu = submit all three model stages as parallel Slurm jobs" >&2
        exit 2
        ;;
esac

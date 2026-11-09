#!/bin/bash
# SLURM job for fine-tuning the shared generator. From the repo root:
#   sbatch scripts/train_generator.sbatch
#   sbatch --gres=gpu:a40:1 scripts/train_generator.sbatch     # another GPU model (command line wins)
# Written for NTU EEE cluster02: one GPU model must be named, and CPUs/RAM come with the GPU
# (--mem/--cpus-per-task are not allowed on GPU jobs). Other clusters: add --partition/--account.
# Re-submitting the same job continues from models/generator/bart-base/last (--resume).
#SBATCH --job-name=evrag-bart
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail   # no -u: conda's activate scripts read unset variables
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Environment: a site file if present (sets caches/TMPDIR and activates the env; not committed),
# otherwise the conda env created from requirements-generator.txt.
SITE_ENV="${EVRAG_SITE_ENV:-local/cluster_env.sh}"
if [ -f "$SITE_ENV" ]; then
  source "$SITE_ENV"
else
  source "${EVRAG_ENV_ACTIVATE:-$HOME/miniconda3/bin/activate}" "${EVRAG_ENV_NAME:-evrag-gen}"
fi

export HF_HOME="${HF_HOME:-$PWD/models/hf_cache}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv

# The data must exist before the job starts if compute nodes have no internet:
#   python -m evrag.data.generator_data      (run once on a login node)
if [ ! -s data/generator/train.jsonl ]; then
  python -m evrag.data.generator_data
fi

OUT="${EVRAG_GENERATOR_OUT:-models/generator/bart-base}"
RESUME=""
if [ -f "$OUT/last/trainer_state.json" ]; then RESUME="--resume"; fi

python scripts/train_generator.py \
  --model "${EVRAG_BASE_MODEL:-facebook/bart-base}" \
  --out "$OUT" \
  --epochs 3 --batch-size 16 --lr 5e-5 --bf16 \
  --eval-every 2000 --log-every 50 $RESUME

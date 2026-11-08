#!/bin/bash
# SLURM job for fine-tuning the shared generator. Fill in the partition/account lines for your
# cluster, then from the repo root:   sbatch scripts/train_generator.sbatch
# Re-submitting the same job continues from models/generator/bart-base/last (--resume).
#SBATCH --job-name=evrag-bart
#SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate the environment created from requirements-generator.txt (edit to match yours).
source "${EVRAG_ENV_ACTIVATE:-$HOME/miniconda3/bin/activate}" "${EVRAG_ENV_NAME:-evrag-gen}"

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

#!/bin/sh
# Basic RAG v1: both routes on dev and test, then metrics and the paired summary (results/basic_v1).
set -eu
cd "$(dirname "$0")/.."
export EVRAG_RESULTS_DIR="${EVRAG_RESULTS_DIR:-results/basic_v1}"
export EVRAG_GENERATOR_DIR="${EVRAG_GENERATOR_DIR:-$PWD/models/generator/bart-base/best}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [ -x "$PWD/.venv/bin/python" ]; then
    python_bin="${PYTHON_BIN:-$PWD/.venv/bin/python}"
else
    python_bin="${PYTHON_BIN:-python3}"
fi
for split in dev test; do
    for method in nlp_only basic_rag; do
        "$python_bin" -u -m evrag.evaluation.run_experiment --method "$method" --split "$split"
    done
    "$python_bin" -m evrag.evaluation.metrics --split "$split" \
        --runs "${EVRAG_RESULTS_DIR}/runs/nlp_only_${split}.jsonl" "${EVRAG_RESULTS_DIR}/runs/basic_rag_${split}.jsonl" \
        --out "${EVRAG_RESULTS_DIR}/metrics_${split}.csv"
done

"$python_bin" scripts/summarize_basic.py

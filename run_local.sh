#!/bin/sh
set -eu
cd "$(dirname "$0")"
export EVRAG_RESULTS_DIR="${EVRAG_RESULTS_DIR:-results/basic_v1}"
export EVRAG_CONFIG="${EVRAG_CONFIG:-$PWD/configs/basic_rag_v1.yaml}"
export EVRAG_GENERATOR_DIR="${EVRAG_GENERATOR_DIR:-$PWD/models/generator/bart-base/best}"
if [ -d "$PWD/models/all-MiniLM-L6-v2" ]; then
    export EVRAG_EMBEDDING_MODEL="${EVRAG_EMBEDDING_MODEL:-$PWD/models/all-MiniLM-L6-v2}"
fi
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
    "$python_bin" -m evrag.evaluation.basic_metrics --split "$split" \
        --runs "${EVRAG_RESULTS_DIR}/runs/nlp_only_${split}.jsonl" "${EVRAG_RESULTS_DIR}/runs/basic_rag_${split}.jsonl" \
        --out "${EVRAG_RESULTS_DIR}/metrics_${split}.csv"
done

"$python_bin" scripts/summarize_basic.py

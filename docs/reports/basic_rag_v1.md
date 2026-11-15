# Basic RAG v1

## Key files

- src/evrag/methods/basic_rag.py connects retrieval and generation.
- src/evrag/methods/nlp_only.py answers the question with the same BART alone.
- src/evrag/retriever.py loads the corpus and caches the embeddings.
- src/evrag/generator.py is the shared BART inference code.
- src/evrag/evaluation/run_experiment.py runs a method over a split and saves per-question JSONL.
- src/evrag/evaluation/metrics.py is E's shared evaluation code (PR #29).
- scripts/summarize_basic.py checks the outputs, evidence text and input lengths, and writes the paired comparison and source tables.
- scripts/run_basic_v1.py runs both methods on the full dev and test splits and evaluates them.

## Run

Create a Python 3.11 virtual environment in the project root and install the dependencies (install the torch build that fits your device). The script overwrites the run and metric files of the same name.

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install torch
    pip install -r requirements.txt -r requirements-generator.txt
    python scripts/run_basic_v1.py

## Explanation

This experiment connects Nam's (A) WikiDoc data (data/processed, see data/README.md), Jojo's shared BART generator, Basic RAG, and E's evaluation code. The corpus has 512 passages, each containing at most 150 whitespace-delimited words. The dataset contains 350 questions with 175 each in dev and test.

NLP-only passes the question directly to the fine-tuned BART checkpoint. Basic RAG uses all-MiniLM-L6-v2 embeddings and normalized FAISS IndexFlatIP search to retrieve five candidates. It selects up to four passages (retrieval.context_k in configs/default.yaml) in rank order while checking the complete BART input against its 1,024-token budget. Both routes use the same generator and decoding parameters. Empty retrieval produces answer=null and abstained=true without invoking BART. Nonempty retrieval invokes BART without an additional sufficiency gate.

Automatic correctness uses E's token-F1 threshold of 0.5. No human labels were applied. Evidence support uses a lexical heuristic, scored against the contexts actually supplied to BART. Evidence support for NLP-only is N/A. Only the aggregate all group is produced when scenario labels are absent.

All questions have gold answers, so abstention recall is N/A. Three normalized questions occur in both splits. The existing checkpoint may overlap the corpus in its training data. The local checkpoint archive hash differs from the published Release archive hash, but its model.safetensors is identical to the one in the Release archive (checked 2026-11-12). These limitations are recorded in results/basic_v1/run_manifest.json. Parameters are fixed for this draft-data comparison.

scripts/run_basic_v1.py uses the BART checkpoint in models/generator/bart-base/best (downloaded from the GitHub Release on first use); EVRAG_GENERATOR_DIR selects another checkpoint directory. sentence-transformers downloads MiniLM. Every setting is read from configs/default.yaml.

The results/basic_v1/runs directory contains the four raw run files. metrics_dev.csv and metrics_test.csv contain automatic metrics. paired_comparison.csv aligns both answers for every question and reports individual gold-passage recall and complete-gold retrieval. context_sources.csv links actual generator contexts to the corpus metadata. Source fields are inherited from the supplied data and may not identify a specific original webpage. Empty CSV metrics mean the measure is undefined for these inputs.

The metric named recall_at_k measures whether any gold passage was retrieved. For multi-evidence questions, also inspect gold_fraction_retrieved, all_gold_retrieved, and context_gold_count in paired_comparison.csv. retrieved_docs contains all five candidates, whereas meta.context_doc_ids lists the passages selected for generation.

The changes add Basic RAG and NLP-only methods, progress logging with flushed run output, aggregate evaluation without scenario labels, and paired-output auditing.

The experiment uses the shared configs/default.yaml; EVRAG_RESULTS_DIR selects the results folder (results/basic_v1 by default).

Data is credited to Nam / Illgetgood (A). Evaluation uses the shared metrics.py by Yuxuan / yx-yan (PR #29), including the scenario-free grouping and evidence-support fixes proposed by Yiwen Xu. The shared BART generator and checkpoint are credited to Jojo082099.

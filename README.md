# Evidence-aware RAG

**When should a RAG agent answer, retrieve again, or abstain?** — EE6405 NLP group project.

> **New to the team? Start here**
> 1. Read [docs/ROLES.md](docs/ROLES.md) and find your letter (A–E). The project idea is summarised in [docs/proposal.md](docs/proposal.md).
> 2. Follow **Setup** below and check that `pytest` passes.
> 3. Open the [issues](../../issues), filter by your `role:` label, and assign this week's issue to yourself.
> 4. Work on a branch and open a PR, following [CONTRIBUTING.md](CONTRIBUTING.md). Key dates are in [docs/timeline.md](docs/timeline.md).

We compare three response policies on one shared retriever:

| # | Method | Policy | Owner |
|---|--------|--------|-------|
| 1 | Standard RAG | retrieve top-k → answer | B |
| 2 | Threshold RAG | abstain if relevance score < threshold (tuned on dev) | C |
| 3 | Evidence-aware RAG | judge evidence sufficiency → answer / abstain | D |

> **Rebuilding (2026-11-08).** The team dropped pretrained LLMs in favour of more basic NLP methods
> (possibly with a decoder), and is preparing a more specific medical dataset. The LLM wrapper, the
> LLM-based methods and the general datasets were removed; they are in git history (last at commit
> `96ad2ba`). Only the interface, the retriever and the `dummy` method remain.

Data (A): new medical dataset in preparation, in three evidence scenarios — A sufficient, B missing, C distracting. See [data/README.md](data/README.md).
Evaluation & demo (E): answer correctness, selective accuracy, coverage, abstention recall, over-refusal, unsupported-answer rate, latency.

## Setup

```bash
conda create -n evrag python=3.11 -y && conda activate evrag
pip install -r requirements.txt
pip install -e .
pytest                 # contract + retriever tests
```

## Run

`run_experiment` needs the new dataset set in `configs/default.yaml`.

```bash
python -m evrag.evaluation.run_experiment --method dummy --split dev --limit 20
streamlit run app/app.py
```

The demo answers with the shared generator (no evidence, pasted evidence, or retrieval once the
corpus is set) and shows how much each input word or evidence sentence contributed to the answer,
with occlusion, LIME and SHAP side by side (`src/evrag/explain.py`). It needs torch and
`transformers` as well (`requirements-generator.txt`).

## Layout

```
configs/default.yaml          frozen shared settings (embeddings, top_k, data paths, seed)
data/                         datasets; see data/README.md
src/evrag/schema.py           RAGOutput — the interface every method returns
src/evrag/methods/            base.py (protocol + registry), dummy, threshold_rag (stub)
src/evrag/retriever.py        shared retriever (sentence-transformers + FAISS/numpy)
src/evrag/generator.py        shared answer generator (BART-base, docs/generator.md)
src/evrag/explain.py          input-token attributions for the demo (occlusion, LIME, SHAP)
src/evrag/evaluation/         run_experiment → results/runs/*.jsonl → metrics → plots
scripts/                      one-off helpers (b_download_model)
notebooks/                    <role>_<topic>.ipynb
results/                      raw outputs per experiment; see results/README.md
docs/                         roles, timeline, proposal
app/app.py                    Streamlit demo with token attributions
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## Basic RAG v1 comparison

See [the bilingual guide](BASIC_RAG_GUIDE_ZH_EN.md) and [the results](results/basic_v1/RESULTS_ZH_EN.md).
Run `./run_local.sh` after installing dependencies and obtaining generator weights.
The experiment uses configs/basic_rag_v1.yaml and data/basic_v1; its metrics entry point is evrag.evaluation.basic_metrics.

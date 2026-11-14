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

The repository includes an asthma, diabetes, and hypertension passage corpus in `data/corpus.jsonl`.
The interactive demo defaults to a graph RAG method that uses the RoBERTa-based
`sentence-transformers/all-distilroberta-v1` model for semantic retrieval, then expands results
through disease and shared-term links. It returns matching source sentences instead of generating
unsupported medical claims. Corpus embeddings are cached under `models/retrieval`.

The question dataset and graph corpus paths are set in `configs/default.yaml`. See [data/README.md](data/README.md).

## Setup

```bash
conda create -n evrag python=3.11 -y && conda activate evrag
pip install -r requirements.txt
pip install -e .
pytest                 # contract + retriever tests
```

## Run

```bash
streamlit run app/app.py
```

Enter a question in the app and select **Ask**. The result includes an extractive answer, graph
connections, and the supporting passages. The first question downloads the embedding model and may
take longer; subsequent starts reuse the model and cached corpus embeddings. The `dummy` method
remains available for the interface contract test only.

## Layout

```
configs/default.yaml          frozen shared settings (embeddings, top_k, data paths, seed)
data/                         datasets; see data/README.md
src/evrag/schema.py           RAGOutput — the interface every method returns
src/evrag/methods/            base.py (protocol + registry), graph_rag, dummy, threshold_rag (stub)
src/evrag/retriever.py        shared retriever (sentence-transformers + FAISS/numpy)
src/evrag/evaluation/         run_experiment → results/runs/*.jsonl → metrics → plots
scripts/                      one-off helpers (b_download_model)
notebooks/                    <role>_<topic>.ipynb
results/                      raw outputs per experiment; see results/README.md
docs/                         roles, timeline, proposal
app/app.py                    Streamlit demo
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

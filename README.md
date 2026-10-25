# Evidence-aware RAG

**When should a RAG agent answer, retrieve again, or abstain?** — EE6405 NLP group project.

> **New to the team? Start here**
> 1. Read [docs/ROLES.md](docs/ROLES.md) and find your letter (A–E). The project idea is summarised in [docs/proposal.md](docs/proposal.md).
> 2. Follow **Setup** below and check that `pytest` passes.
> 3. Open the [issues](../../issues), filter by your `role:` label, and assign this week's issue to yourself.
> 4. Work on a branch and open a PR, following [CONTRIBUTING.md](CONTRIBUTING.md). Key dates are in [docs/timeline.md](docs/timeline.md).

We compare three response policies on one shared retriever and one LLM:

| # | Method | Policy | Owner |
|---|--------|--------|-------|
| 1 | Standard RAG | retrieve top-k → answer | B |
| 2 | Threshold RAG | abstain if relevance score < threshold (tuned on dev) | C |
| 3 | Evidence-aware RAG | LLM judges evidence sufficiency → answer / abstain (optional one retry) | D |

Data (A): SQuAD 2.0-based fixed corpus, three evidence scenarios — A sufficient, B missing, C distracting.
Evaluation & demo (E): answer correctness, selective accuracy, coverage, abstention recall, over-refusal, unsupported-answer rate, latency.

## Setup

```bash
conda create -n evrag python=3.11 -y && conda activate evrag
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # add your LLM key
pytest                 # contract tests
```

## Run

```bash
python -m evrag.evaluation.run_experiment --method dummy --split dev --limit 20
streamlit run app/app.py
```

## Layout

```
configs/default.yaml          frozen shared settings (embeddings, top_k, LLM, seed)
data/processed/               dataset.csv + corpus.jsonl (committed, frozen after Week 1)
src/evrag/schema.py           RAGOutput — the interface every method returns
src/evrag/methods/            base.py (protocol + registry), baseline / threshold_rag / evidence_rag
src/evrag/retriever.py, llm.py  shared backend
src/evrag/evaluation/         run_experiment → results/runs/*.jsonl → metrics → plots
app/app.py                    Streamlit demo
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

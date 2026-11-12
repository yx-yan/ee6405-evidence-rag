# Roles & tasks

Find your letter, read your section, then take your issue for the current week (label `role:<letter>`).

| Role | Goal | Key output | Depends on |
|------|-----------|------------|------------|
| **A · Data** | Build a fixed, audited retrieval corpus and evaluation set from SQuAD 2.0, covering the three evidence scenarios. | `dataset.csv` (question IDs, gold answers, scenario, dev/test split) and `corpus.jsonl`, used by **everyone**. | Nothing to start. Agree the `dataset.csv` columns with E. |
| **B · Baseline** | Build the shared retriever and LLM wrapper, plus Standard RAG. This is the foundation C and D reuse. | `retriever.search()`, `llm.generate()` and the `baseline` method, used by C, D and E. | `corpus.jsonl` from A (use a toy corpus until it exists); settings in `configs/default.yaml`. |
| **C · Threshold RAG** | Add a simple control: abstain when the retrieval relevance score is below a threshold tuned on dev. | The `threshold` method and `results/threshold_sweep.csv` (E draws Figure 2 from it). | B's retriever scores; A's dev split. |
| **D · Evidence-aware agent** | Have the LLM judge whether the retrieved excerpts support an answer (SUFFICIENT / INSUFFICIENT) before answering or abstaining. | The `evidence` method with judge verdict, supporting snippet and `retried` flag in `meta`; case studies. | B's retriever and LLM wrapper (same model as everyone else). |
| **E · Evaluation & demo** | Own the shared interface and turn every method's outputs into metrics, figures and a Streamlit demo. | The interface contract, `results/metrics.csv`, Figures 1–3, the demo; integration review of PRs. | Every method's `results/runs/*.jsonl`; A's `dataset.csv`. |

**Interface contract**: every method returns `RAGOutput` (see `src/evrag/schema.py`). Don't change it without E.

## A · Data

**Goal:** Build a fixed, audited retrieval corpus and evaluation set from SQuAD 2.0, covering the three evidence scenarios.

**You own:** `src/evrag/data/build_corpus.py`, `data/README.md`, `data/processed/dataset.csv`, `data/processed/corpus.jsonl`, `notebooks/A_data_audit.ipynb`

**You need:** Nothing to start. Agree the `dataset.csv` columns with E.  
**Others need from you:** `dataset.csv` (question IDs, gold answers, scenario, dev/test split) and `corpus.jsonl`, used by **everyone**.

**Week 1 · Establish baseline** (Oct 26 – Nov 1)

- [ ] Agree the `dataset.csv` columns with E (by **Oct 28**)
- [ ] Choose the SQuAD 2.0 subset (which articles/titles) and how to split dev/test
- [ ] Ship a first small version (~20 questions incl. all three scenarios) so the team can run 20 end-to-end cases

**Week 2 · Compare methods** (Nov 2 – Nov 8)

- [ ] Scale to 150–300 questions total
- [ ] Scenario **B** (missing): withhold gold passages / use unanswerable questions, and check the answer is absent from the **whole** corpus
- [ ] Scenario **C** (distracting): add related passages that lack the requested fact (optional small contradictory subset)
- [ ] Deduplicate; check no dev/test leakage
- [ ] Manually audit ≥10 items per scenario; write notes in `data/README.md`
- [ ] Freeze the data: commit and tag `data-v1`

**Week 3 · Package results** (Nov 9 – Nov 15)

- [ ] Dataset statistics table for the slides (counts per scenario × split)
- [ ] Help with error analysis: was a failure caused by the retriever or by missing evidence?

**Done when:** Frozen `data/processed/` committed; every "no answer" label holds against the full corpus; audit notes written.

## B · Baseline

**Goal:** Build the shared retriever and LLM wrapper, plus Standard RAG. This is the foundation C and D reuse.

**You own:** `src/evrag/retriever.py`, `src/evrag/llm.py`, `src/evrag/methods/baseline.py`

**You need:** `corpus.jsonl` from A (use a toy corpus until it exists); settings in `configs/default.yaml`.  
**Others need from you:** `retriever.search()`, `llm.generate()` and the `baseline` method, used by C, D and E.

**Week 1 · Establish baseline** (Oct 26 – Nov 1)

- [ ] Pick the single LLM with the team (cost, rate limits, access) and fill `configs/default.yaml`
- [ ] Implement `llm.generate(prompt)` (key from `.env`)
- [ ] Implement `retriever.search(query, top_k)` with sentence-transformers + FAISS/cosine
- [ ] Register `baseline` (retrieve top-k → prompt → answer); `pytest` passes
- [ ] Run the first 20 end-to-end cases with A's starter data

**Week 2 · Compare methods** (Nov 2 – Nov 8)

- [ ] Build and cache the index on A's frozen corpus
- [ ] Report retrieval recall@k of gold passages (separates retriever failure from missing evidence)
- [ ] Full dev + test runs → `results/runs/baseline_*.jsonl`
- [ ] Support C and D in using the shared API (no copy-pasted retrieval code)

**Week 3 · Package results** (Nov 9 – Nov 15)

- [ ] Baseline prompt + 2–3 failure cases for the slides
- [ ] Clean up and add docstrings for retriever/llm

**Done when:** C and D call your modules directly; baseline runs on the full test split without errors.

## C · Threshold RAG

**Goal:** Add a simple control: abstain when the retrieval relevance score is below a threshold tuned on dev.

**You own:** `src/evrag/methods/threshold_rag.py`, `notebooks/C_threshold_tuning.ipynb`

**You need:** B's retriever scores; A's dev split.  
**Others need from you:** The `threshold` method and `results/threshold_sweep.csv` (E draws Figure 2 from it).

**Week 1 · Establish baseline** (Oct 26 – Nov 1)

- [ ] Decide the relevance score (top-1 cosine, mean of top-k, ... ) and justify it
- [ ] Prototype `threshold_rag.py` against `DummyMethod`-style scores; register it; `pytest` passes

**Week 2 · Compare methods** (Nov 2 – Nov 8)

- [ ] Switch to B's real retriever
- [ ] Sweep thresholds on **dev only**; choose an operating point with a stated rule
- [ ] Optional ablation over `top_k`
- [ ] Save the sweep → `results/threshold_sweep.csv`; run once on test → `results/runs/threshold_test.jsonl`

**Week 3 · Package results** (Nov 9 – Nov 15)

- [ ] Interpretation: why high similarity ≠ sufficient evidence; over-refusal examples
- [ ] Slides for method 2

**Done when:** Threshold chosen on dev only and documented; sweep CSV + test run committed.

## D · Evidence-aware agent

**Goal:** Have the LLM judge whether the retrieved excerpts support an answer (SUFFICIENT / INSUFFICIENT) before answering or abstaining.

**You own:** `src/evrag/methods/evidence_rag.py`, `configs/prompts/ (new)`, `notebooks/D_judge_validation.ipynb`

**You need:** B's retriever and LLM wrapper (same model as everyone else).  
**Others need from you:** The `evidence` method with judge verdict, supporting snippet and `retried` flag in `meta`; case studies.

**Week 1 · Establish baseline** (Oct 26 – Nov 1)

- [ ] Draft the sufficiency prompt: output SUFFICIENT/INSUFFICIENT + a supporting snippet (structured, easy to parse)
- [ ] Robust parsing (malformed output → treat as INSUFFICIENT and log it)
- [ ] Register `evidence`; `pytest` passes

**Week 2 · Compare methods** (Nov 2 – Nov 8)

- [ ] Integrate with B's retriever + LLM
- [ ] Manually validate the judge on ≥30 dev items; report agreement
- [ ] Fill `meta`: `verdict`, `snippet`, `retried`
- [ ] **Optional, only once the above works**: one query rewrite + re-retrieval when the evidence is ambiguous
- [ ] Full dev + test runs → `results/runs/evidence_*.jsonl`

**Week 3 · Package results** (Nov 9 – Nov 15)

- [ ] Case studies: one hallucination prevented, one wrong refusal, one retry that helped (if implemented)
- [ ] Cost/latency discussion; slides for method 3

**Done when:** Judge agreement on the manual sample reported; `meta` populated; test run committed.

## E · Evaluation & demo

**Goal:** Own the shared interface and turn every method's outputs into metrics, figures and a Streamlit demo.

**You own:** `src/evrag/schema.py`, `src/evrag/methods/base.py`, `src/evrag/evaluation/`, `src/evrag/explain.py`, `app/app.py`, `tests/`

**You need:** Every method's `results/runs/*.jsonl`; A's `dataset.csv`.  
**Others need from you:** The interface contract, `results/metrics.csv`, Figures 1–3, the demo; integration review of PRs.

**Week 1 · Establish baseline** (Oct 26 – Nov 1)

- [ ] Freeze the interface (`schema.py`, `base.py`, contract test) by **Oct 28**
- [ ] Agree the `dataset.csv` columns with A
- [ ] `run_experiment` works end-to-end with `dummy`
- [ ] Review the first method PRs for contract compliance

**Week 2 · Compare methods** (Nov 2 – Nov 8)

- [ ] Implement all metrics (EM / token F1, correctness, selective accuracy, coverage, abstention recall, over-refusal, unsupported-answer rate, latency mean/p95), stratified by scenario
- [ ] Manual adjudication of a stratified subset of answers / support judgments
- [ ] Run all methods on test → `results/metrics.csv`

**Week 3 · Package results** (Nov 9 – Nov 15)

- [ ] Figure 1 (stacked outcomes), Figure 2 (accuracy vs coverage), Figure 3 (latency vs reliability)
- [ ] Streamlit demo: method switch, answer/abstain, evidence
- [x] Demo shows each input token's contribution to the answer (occlusion, LIME, SHAP; Week 9 requirement)
- [ ] Assemble slides; coordinate video recording; code freeze + README reproduction steps

**Done when:** One command reproduces `metrics.csv` and all figures from the committed runs.

## Everyone

- [ ] Contribute to the **interpretation** in the slides, not only your own code
- [ ] **Appear in the video** (≤10 min, no AI voice)
- [ ] Write your part of the **signed group reflection** (what you did, with commit/PR links)
- [ ] Keep **shareable links** for any AI-tool chats you used (AI disclosure form)
- [ ] Review at least one teammate's PR

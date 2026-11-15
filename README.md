# Evidence-aware RAG

**When should a medical RAG system answer, retrieve again, or abstain?** — EE6405 NLP group project.

A retrieval-augmented QA system answers from whatever it retrieves. Our generator, a BART-base
model we fine-tuned on medical QA, never says "I don't know": when the evidence lacks the answer it
still copies from it. We compare five ways of using the same retriever and generator, from no
retrieval to an evidence-aware policy that checks whether the evidence is sufficient, retrieves
again with an expanded query when it is not, and abstains when it still is not. No large language
model is used anywhere.

## Datasets

Three single-disease knowledge bases from three sources ([data/README.md](data/README.md)):

| Set | Source | Passages | Test questions: answerable / missing evidence / out of domain |
|-----|--------|---------:|------------------------------------------------------------:|
| coronavirus | [COVID-QA](https://github.com/deepset-ai/COVID-QA) (CORD-19 papers) | 922 | 106 / 25 / 50 |
| cancer | [MedQuAD](https://github.com/abachaa/MedQuAD), cancer.gov | 3,844 | 125 / 25 / 50 |
| neuro | MedQuAD, ninds.nih.gov | 1,333 | 125 / 25 / 50 |

Each set is split into dev (for tuning) and test. **Missing evidence** questions have their gold
passages removed from the corpus; **out of domain** questions come from the other two sets.

## Methods

All five use the same BART generator, and the dense routes share one MiniLM retriever (FAISS, top
5); only the evidence and the decision to abstain differ ([src/evrag/methods/](src/evrag/methods/)).

| Method | Policy | Owner |
|--------|--------|-------|
| `nlp_only` | closed book: the generator answers from the question alone | B |
| `basic_rag` | retrieve top 5 → answer from the passages that fit 1,024 tokens | B |
| `graph_rag` | term/passage graph: direct term matches plus one hop through shared rare terms → answer | C |
| `threshold_rag` | `basic_rag`, but abstain when the top-1 retrieval score is below τ | E |
| `evidence_rag` | two evidence signals → answer / retrieve again (Rocchio query expansion) / abstain | E |

`evidence_rag` signals: **s1**, the top-1 dense retrieval score, and **s2**, the IDF-weighted share
of the question's content words found in the top passage. Both thresholds and τ are tuned per
dataset on dev by maximising Youden's J (abstention recall − over-refusal rate)
([src/evrag/evaluation/tune.py](src/evrag/evaluation/tune.py)).

We also compare the retrievers alone: TF-IDF, our own BM25, MiniLM and the graph retriever
(Recall@1/5, MRR; [retrieval_eval.py](src/evrag/evaluation/retrieval_eval.py)).

## Metrics

[src/evrag/evaluation/metrics.py](src/evrag/evaluation/metrics.py): coverage, answer correctness
(token F1 ≥ 0.5, with a 95% bootstrap interval), selective accuracy, reliability (correct answers
plus correct abstentions), abstention recall, over-refusal rate, unsupported-answer rate, token F1,
ROUGE-L, gold token recall, BLEU-4, METEOR, recall@k and latency, overall and per scenario, plus a
paired bootstrap between methods.

## Results (test)

Each test set contains 43% unanswerable questions. *Correct* means token F1 ≥ 0.5 against a gold
answer. *Reliability* is the share of all questions with a correct answer or a correct abstention.
*Abstain* is the share of unanswerable questions the method declined. *Over-refusal* is the share
of answerable questions it declined.

| Set | Method | Coverage | Correct | Reliability | Abstain (unanswerable) | Over-refusal | Gold token recall |
|-----|--------|---------:|--------:|------------:|-----------------------:|-------------:|------------------:|
| coronavirus | nlp_only | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| | basic_rag | 1.00 | 0.06 | 0.06 | 0.00 | 0.00 | 0.38 |
| | graph_rag | 0.96 | **0.13** | 0.17 | 0.11 | 0.00 | **0.55** |
| | threshold_rag | 0.53 | 0.03 | **0.30** | **0.67** | 0.33 | 0.39 |
| | evidence_rag | 0.61 | 0.06 | **0.30** | 0.60 | 0.24 | 0.45 |
| cancer | nlp_only | 1.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.17 |
| | basic_rag | 1.00 | 0.21 | 0.21 | 0.00 | 0.00 | 0.36 |
| | graph_rag | 0.96 | 0.19 | 0.23 | 0.11 | 0.00 | 0.31 |
| | threshold_rag | 0.71 | 0.20 | **0.48** | **0.73** | 0.02 | 0.36 |
| | evidence_rag | 0.74 | **0.22** | **0.48** | 0.69 | 0.01 | 0.37 |
| neuro | nlp_only | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 |
| | basic_rag | 1.00 | **0.34** | 0.34 | 0.00 | 0.00 | 0.47 |
| | graph_rag | 1.00 | 0.31 | 0.32 | 0.01 | 0.00 | 0.46 |
| | threshold_rag | 0.72 | 0.33 | **0.59** | 0.68 | 0.04 | 0.48 |
| | evidence_rag | 0.70 | 0.32 | **0.59** | **0.71** | 0.06 | 0.47 |

What the results show:

- **Abstaining adds about 0.25 reliability on every set without lowering correctness.** Both
  abstaining methods catch 82–100% of out-of-domain questions.
- **Missing evidence is the open problem.** 72–96% of missing-evidence questions are still
  answered, because a passage on the same disease stays in the corpus and scores almost as high.
- **`evidence_rag` matches `threshold_rag`; it does not beat it.** Their reliability is equal on all
  three sets (paired bootstrap). Retrieving again with an expanded query rarely recovers a missed
  gold passage.
- **On coronavirus, read gold token recall instead of correctness.** COVID-QA gold answers are
  short spans inside longer generated answers, so token F1 ≥ 0.5 is near zero for every method.
  By gold token recall, `graph_rag` is best, matching its stronger lexical retrieval on this set.
- **Closed-book generation fails.** `nlp_only` answers almost nothing correctly; the generator
  needs the retrieved evidence.

All tables and figures: [notebooks/final_results.ipynb](notebooks/final_results.ipynb) (with
outputs) and [results/](results/README.md).

## Reproduce

```bash
conda create -n evrag python=3.11 -y && conda activate evrag
pip install torch                                     # CUDA, MPS (Apple) or CPU build
pip install -r requirements.txt -r requirements-generator.txt && pip install -e .
pytest

python -m evrag.data.disease_sets                     # (optional) rebuild data/processed/<set>/ from the sources
python -m evrag.data.unanswerable                     # evaluation versions with unanswerable questions (committed)
python -m evrag.data.train_overlap                    # flag answers seen in generator training (committed)

python -m evrag.evaluation.tune                       # dev thresholds -> results/final/thresholds.json
python -m evrag.evaluation.retrieval_eval             # retriever comparison
python scripts/run_final.py                           # all methods, datasets and splits (~1.5 h on an M-series Mac; resumable)
python scripts/final_metrics.py                       # metrics, paired bootstrap, risk-coverage
jupyter nbconvert --to notebook --execute --inplace notebooks/final_results.ipynb

streamlit run app/app.py                              # demo
```

The generator weights (0.5 GB) are included in the submission archive under
`models/generator/bart-base/best/`; in a git clone they are downloaded from the GitHub Release on
first use (training: `scripts/train_generator.py`). A single run:
`python -m evrag.evaluation.run_experiment --dataset cancer --method evidence_rag --split dev --limit 20`.

The demo lets you pick a dataset and a method, shows the decision trace (both signals against
their thresholds, the expanded query, the passages and scores) and then which input words the
answer relied on (occlusion, LIME, SHAP). Example buttons cover an answerable question, a retry,
missing evidence, an out-of-domain question and a distractor passage.

## Layout

```text
configs/default.yaml          shared settings: retriever, generator, seed, datasets by name
data/                         datasets; see data/README.md
src/evrag/schema.py           RAGOutput, the interface every method returns
src/evrag/methods/            base.py (protocol + registry) and the five methods
src/evrag/retriever.py        shared dense retriever (sentence-transformers + FAISS), one index per dataset
src/evrag/generator.py        shared answer generator (BART-base)
src/evrag/explain.py          input-token attributions for the demo (occlusion, LIME, SHAP)
src/evrag/data/               dataset builders: disease sets, unanswerable scenarios, training overlap
src/evrag/evaluation/         run_experiment, metrics, threshold tuning, retriever comparison
scripts/                      run_final.py, final_metrics.py, generator training, WikiDoc appendix run
notebooks/final_results.ipynb results with outputs
results/                      raw outputs and tables; see results/README.md
app/app.py                    Streamlit demo
```

The earlier WikiDoc run ([results/basic_v1/](results/basic_v1/RESULTS.md)) is kept as an appendix: the generator saw WikiDoc in
training, so it is not a main evaluation set.

## Team

| Role | Member | Work |
|------|--------|------|
| A | Nguyen Khanh Nam ([Illgetgood](https://github.com/Illgetgood)) | data: WikiDoc and the three disease sets |
| B | Xu Yiwen ([elenaxu610](https://github.com/elenaxu610)) | shared retriever and basic RAG |
| C | Nur Hapinder ([Victoria0706](https://github.com/Victoria0706)) | Graph RAG and the retriever comparison (TF-IDF, BM25, MiniLM, graph) |
| D | Zhan Luming ([Jojo082099](https://github.com/Jojo082099)) | shared generator (BART-base fine-tuning) |
| E | Yan Yuxuan ([yx-yan](https://github.com/yx-yan)) | evaluation, threshold and evidence-aware RAG, demo, integration |

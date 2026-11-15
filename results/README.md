# Results

| Folder / file | Content | Written by |
|---------------|---------|------------|
| `final/thresholds.json` | Abstention thresholds per dataset, tuned on dev (Youden's J). | `python -m evrag.evaluation.tune` |
| `final/signals/` | evidence_rag's signals for every question (both retrievals), per dataset and split. | `python -m evrag.evaluation.tune` |
| `final/retrieval.csv`, `final/recall_curve.csv` | TF-IDF / BM25 / MiniLM / graph retrieval on answerable test questions. | `python -m evrag.evaluation.retrieval_eval` |
| `final/runs/<dataset>/<method>_<split>.jsonl` | Raw method outputs (`RAGOutput` rows). threshold_rag is derived from basic_rag. | `scripts/run_final.py`, `scripts/final_metrics.py` |
| `final/metrics.csv`, `final/metrics_unseen.csv` | Metrics per dataset, split, method and scenario (all questions / without answers seen in generator training). | `scripts/final_metrics.py` |
| `final/paired.csv` | Paired bootstrap, evidence_rag against each baseline (test). | `scripts/final_metrics.py` |
| `final/risk_coverage.csv` | basic_rag coverage and selective accuracy as the score threshold sweeps. | `scripts/final_metrics.py` |
| `figures/` | Figures from the results notebook. | `notebooks/final_results.ipynb` |
| `basic_v1/` | Appendix: the first basic_rag vs nlp_only run on WikiDoc (seen in generator training, so not a main evaluation set). | `scripts/run_basic_v1.sh` |

Earlier LLM screens and MedQuAD trials were removed on 2026-11-08 and are in git history.

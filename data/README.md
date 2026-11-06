# Data (owner: A)

| Folder | Content |
|--------|---------|
| `medquad/` | Current knowledge base and A/B/C questions, used by `configs/default.yaml`. See [medquad/README.md](medquad/README.md). |
| `processed/` | MTS-Dialog in the project schema, kept to reproduce the LLM screen. Built from `mts_dialog/` by `scripts/b_prepare_data.py`. |
| `mts_dialog/` | A's MTS-Dialog source and annotated files. The preprocessing notebooks are `notebooks/A_mts_dialog_preprocess*.ipynb`. |
| `raw/` | Local downloads, not committed: MedQuAD clone, `liveqa_med/`, `symptom2disease/`. |

Every evaluation set uses this schema:

| File | Columns / fields |
|------|------------------|
| `corpus.jsonl` | `doc_id, title, text` |
| `dataset.csv` | `question_id, question, gold_answers, scenario (A/B/C), split (dev/test), gold_doc_ids` |

Scenarios: **A** sufficient evidence · **B** missing evidence · **C** distracting evidence.

TODO(A): document the manual audit results.

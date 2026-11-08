# Data (owner: A)

The team is preparing a new, more specific medical dataset (for example heart disease). The general
datasets used earlier (MedQuAD, LiveQA-Med, Symptom2Disease, MTS-Dialog) were removed on 2026-11-08;
they are in git history before that date. Point `data.dataset` and `data.corpus` in
`configs/default.yaml` at the new files once they exist.

| Folder | Content |
|--------|---------|
| `processed/` | The frozen evaluation set (committed). |
| `raw/` | Local downloads, not committed. |

Every evaluation set uses this schema:

| File | Columns / fields |
|------|------------------|
| `corpus.jsonl` | `doc_id, title, text` |
| `dataset.csv` | `question_id, question, gold_answers, scenario (A/B/C), split (dev/test), gold_doc_ids` |

Scenarios: **A** sufficient evidence · **B** missing evidence · **C** distracting evidence.

TODO(A): document the manual audit results.

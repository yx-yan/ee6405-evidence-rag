# Data (owner: A)

Source: SQuAD 2.0 (Rajpurkar, Jia & Liang, 2018) — https://rajpurkar.github.io/SQuAD-explorer/

Built by `python -m evrag.data.build_corpus`:

| File | Columns / fields |
|------|------------------|
| `processed/corpus.jsonl` | `doc_id, title, text` |
| `processed/dataset.csv` | `question_id, question, gold_answers, scenario (A/B/C), split (dev/test), gold_doc_ids` |

Scenarios: **A** sufficient evidence · **B** missing evidence · **C** distracting evidence.

TODO(A): document sampling, deduplication, how "no answer" was verified against the *whole* corpus, and the manual audit results.

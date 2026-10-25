"""Build the fixed retrieval corpus and evaluation set from SQuAD 2.0. Owner: A.

Outputs (committed to git, frozen after Week 1):
  data/processed/corpus.jsonl  — {doc_id, title, text}
  data/processed/dataset.csv   — question_id, question, gold_answers, scenario (A/B/C), split (dev/test), gold_doc_ids

TODO(A): implement; document choices in data/README.md.
"""

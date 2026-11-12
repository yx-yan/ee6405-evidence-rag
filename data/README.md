# Data (owner: A)

The knowledge base and evaluation questions come from WikiDoc patient information
([medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)),
keeping every question/answer pair that mentions asthma, diabetes or hypertension. Each answer is
split into 150-word passages with a 30-word overlap; its question becomes an evaluation question
whose gold answer is the full answer and whose gold documents are its passages. Questions alternate
between dev and test. Built by `notebooks/A_data_preprocess.ipynb`. The general datasets used earlier (MedQuAD,
LiveQA-Med, Symptom2Disease, MTS-Dialog) were removed on 2026-11-08 and are in git history.

| Path | Content |
|------|---------|
| `processed/corpus.jsonl` | 512 passages: `doc_id, article_id, title, section, text, disease, source_url` |
| `processed/dataset.csv` | 350 questions, 175 dev / 175 test: `question_id, question, gold_answers, gold_doc_ids, split` |
| `processed/by_disease/` | The same passages split by the diseases they mention (e.g. `asthma_diabetes.jsonl`) |
| `raw/` | Local downloads, not committed. |

`gold_answers` and `gold_doc_ids` are JSON lists. A dataset may add a `scenario` column; metrics
are then also reported per scenario.

Known limits:

- Every question has a gold answer, so abstention cannot be measured on this set.
- Three questions occur in both splits (after lower-casing and dropping punctuation): "what causes
  gum disease", "what causes tetralogy of fallot", "who is at risk for hepatocellular carcinoma".
- The gold passages hold the gold answer word for word, and a passage's `title` is the first 50
  characters of its question. The retriever encodes `text` only, so the title does not leak.
- Questions mention a target disease, but many are about something else (e.g. gum disease).
- The generator was trained on WikiDoc patient information too, so most passages overlap its
  training data (issue #26).

TODO(A): document the manual audit results.

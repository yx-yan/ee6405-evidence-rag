# Data (owner: A)

The knowledge base and evaluation questions come from WikiDoc patient information
([medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)),
keeping every question/answer pair that mentions asthma, diabetes or hypertension. Each answer is
split into 150-word passages with a 30-word overlap; its question becomes an evaluation question
whose gold answer is the full answer and whose gold documents are its passages. Questions alternate
between dev and test. Built by `python -m evrag.data.wikidoc` ([src/evrag/data/wikidoc.py](../src/evrag/data/wikidoc.py)). The general datasets used earlier (MedQuAD,
LiveQA-Med, Symptom2Disease, MTS-Dialog) were removed on 2026-11-08 and are in git history.

| Path | Content |
|------|---------|
| `processed/corpus.jsonl` | 512 passages: `doc_id, article_id, title, section, text, disease, source_url` |
| `processed/dataset.csv` | 350 questions, 175 dev / 175 test: `question_id, question, gold_answers, gold_doc_ids, split` |
| `processed/by_disease/` | The same passages split by the diseases they mention (e.g. `asthma_diabetes.jsonl`) |
| `raw/` | Local downloads, not committed. |

## Single-disease sets

Three more sets, one disease category and one source each, in the same schema. Built by
`python -m evrag.data.disease_sets` ([src/evrag/data/disease_sets.py](../src/evrag/data/disease_sets.py)),
which downloads COVID-QA and clones MedQuAD into `raw/`.

| Folder | Disease category | Source | Licence | Passages | Questions (dev / test) | MiniLM recall@5 |
|--------|------------------|--------|---------|---------:|-----------------------:|----------------:|
| `processed/coronavirus/` | coronaviruses | [COVID-QA](https://github.com/deepset-ai/COVID-QA): the 39 CORD-19 papers whose title names a coronavirus | Apache 2.0 | 982 | 150 / 150 | 0.41 |
| `processed/cancer/` | cancer | [MedQuAD](https://github.com/abachaa/MedQuAD) cancer.gov (NCI) pages | CC BY 4.0 | 3,918 | 150 / 150 | 0.92 |
| `processed/neuro/` | neurological disorders | MedQuAD ninds.nih.gov (NINDS) pages | CC BY 4.0 | 1,394 | 150 / 150 | 0.99 |

Recall@5 counts questions with at least one gold passage in the top 5 (all 300 questions).

- Papers and answers are cut into the same 150-word passages with a 30-word overlap as WikiDoc.
  Each passage's text starts with its paper title or page focus ("Skin Cancer: ..."), like a
  page heading; WikiDoc passages have no heading.
- COVID-QA gold answers are short spans (median 9 words) copied from the paper; its gold documents
  are the passages holding the span. MedQuAD gold answers are whole page sections (up to 300 words);
  their gold documents are all of the section's passages.
- Evaluation questions skip MedQuAD's `research` type (boilerplate about NCI and NINDS research
  programmes) and answers over 300 words; those passages stay in the corpus. Questions asked about
  several papers or pages are merged into one row with every gold answer and passage. 300 questions
  per set are drawn with the config seed.
- MedQuAD questions are templated from the page focus ("What are the treatments for X ?"), which
  makes retrieval easy; COVID-QA questions were written by annotators against one paper.
- The shared generator was not trained on COVID-QA or MedQuAD, but its MEDIQA-AnS training data
  comes from NIH pages and may overlap the MedQuAD sets.

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

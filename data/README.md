# Data (owner: A)

The main evaluation uses three single-disease knowledge bases, each from its own source, plus
unanswerable questions built from them. WikiDoc, the first knowledge base, is kept only as an
appendix because the generator saw it in training.

| Path | Content |
|------|---------|
| `processed/<set>/corpus.jsonl`, `processed/<set>/dataset.csv` | The disease sets as built from their sources. |
| `processed/<set>/eval/` | The evaluation versions with unanswerable questions, and `train_overlap.csv`. These are the main evaluation sets. |
| `processed/corpus.jsonl`, `processed/dataset.csv`, `processed/by_disease/` | WikiDoc (appendix). |
| `raw/` | Local downloads, not committed. |

`gold_answers` and `gold_doc_ids` are JSON lists. Corpus rows are
`doc_id, article_id, title, section, text, disease, source_url`. Question rows are
`question_id, question, gold_answers, gold_doc_ids, split`; eval sets add `scenario` and
`source_set`, and metrics are then also reported per scenario.

## Single-disease sets

One disease category and one source per set. Built by `python -m evrag.data.disease_sets`
([src/evrag/data/disease_sets.py](../src/evrag/data/disease_sets.py)), which downloads COVID-QA
and clones MedQuAD into `raw/`.

| Folder | Disease category | Source | Licence | Passages | Questions (dev / test) | MiniLM recall@5 |
|--------|------------------|--------|---------|---------:|-----------------------:|----------------:|
| `processed/coronavirus/` | coronaviruses | [COVID-QA](https://github.com/deepset-ai/COVID-QA): the 39 CORD-19 papers whose title names a coronavirus | Apache 2.0 | 982 | 150 / 150 | 0.41 |
| `processed/cancer/` | cancer | [MedQuAD](https://github.com/abachaa/MedQuAD) cancer.gov (NCI) pages | CC BY 4.0 | 3,918 | 150 / 150 | 0.92 |
| `processed/neuro/` | neurological disorders | MedQuAD ninds.nih.gov (NINDS) pages | CC BY 4.0 | 1,394 | 150 / 150 | 0.99 |

Recall@5 counts questions with at least one gold passage in the top 5 (all 300 questions).

- Papers and answers are cut into 150-word passages with a 30-word overlap. Each passage's text
  starts with its paper title or page focus ("Skin Cancer: ..."), like a page heading.
- COVID-QA gold answers are short spans (median 9 words) copied from the paper; its gold documents
  are the passages holding the span. MedQuAD gold answers are whole page sections (up to 300 words);
  their gold documents are all of the section's passages.
- Evaluation questions skip MedQuAD's `research` type (boilerplate about NCI and NINDS research
  programmes) and answers over 300 words; those passages stay in the corpus. Questions asked about
  several papers or pages are merged into one row with every gold answer and passage. 300 questions
  per set are drawn with the config seed.
- MedQuAD questions are templated from the page focus ("What are the treatments for X ?"), which
  makes retrieval easy; COVID-QA questions were written by annotators against one paper.

## Evaluation sets with unanswerable questions

These are the main evaluation sets (`datasets:` in `configs/default.yaml`).
`python -m evrag.data.unanswerable` ([src/evrag/data/unanswerable.py](../src/evrag/data/unanswerable.py))
writes `processed/<set>/eval/{corpus.jsonl,dataset.csv}` without touching the original files.
Each question has a `scenario`, and `source_set` names the set it comes from:

- `answerable`: the set's own questions, unchanged.
- `missing_evidence`: 25 questions per split whose gold passages are removed from the eval corpus;
  their gold answers are emptied. Other passages of the same paper or page stay, so retrieval still
  finds related text that does not hold the answer. Answerable questions whose gold passages were
  all removed (shared COVID-QA chunks) are dropped.
- `out_of_domain`: 25 questions per split from each of the other two sets (same split), gold emptied.

Questions are drawn with the config seed within each split, so dev and test never share one.

| Set | Eval passages (removed) | Answerable dropped | dev: answerable / missing / out-of-domain | test: answerable / missing / out-of-domain |
|-----|------------------------:|-------------------:|------------------------------------------:|-------------------------------------------:|
| coronavirus | 922 (60) | 37 | 107 / 25 / 50 | 106 / 25 / 50 |
| cancer | 3,844 (74) | 0 | 125 / 25 / 50 | 125 / 25 / 50 |
| neuro | 1,333 (61) | 0 | 125 / 25 / 50 | 125 / 25 / 50 |

MiniLM recall@5 on the eval answerable questions: coronavirus 0.41, cancer 0.91, neuro 0.99.

**Overlap with generator training.** `python -m evrag.data.train_overlap` writes
`eval/train_overlap.csv`. An answerable question is `seen` when at least half of the word 8-grams
of a gold answer occur in the generator's training sources (all splits, so this is an upper bound).
MEDIQA-AnS takes its sections from NIH pages, including cancer.gov and NINDS. Seen: coronavirus
0 / 213, cancer 24 / 250, neuro 14 / 250. The results notebook also reports the main table without
them.

## Appendix: WikiDoc

The first knowledge base came from WikiDoc patient information
([medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)).
It keeps every question/answer pair that mentions asthma, diabetes or hypertension. Each answer is
split into 150-word passages with a 30-word overlap. Its question becomes an evaluation question
whose gold answer is the full answer and whose gold documents are its passages. Questions alternate
between dev and test. Built by `python -m evrag.data.wikidoc`
([src/evrag/data/wikidoc.py](../src/evrag/data/wikidoc.py)); results are in
[results/basic_v1/](../results/basic_v1/).

| Path | Content |
|------|---------|
| `processed/corpus.jsonl` | 512 passages |
| `processed/dataset.csv` | 350 questions, 175 dev / 175 test |
| `processed/by_disease/` | The same passages split by the diseases they mention (e.g. `asthma_diabetes.jsonl`) |

Why it is only an appendix:

- The generator was trained on WikiDoc patient information too, so most passages overlap its
  training data (issue #26).
- Every question has a gold answer, so abstention cannot be measured on it.
- Three questions occur in both splits (after lower-casing and dropping punctuation): "what causes
  gum disease", "what causes tetralogy of fallot", "who is at risk for hepatocellular carcinoma".
- Many questions mention a target disease but are about something else (e.g. gum disease).

Earlier general datasets (MedQuAD as a whole, LiveQA-Med, Symptom2Disease, MTS-Dialog) were removed
on 2026-11-08 and are in git history.

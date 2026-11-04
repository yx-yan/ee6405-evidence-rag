# MedQuAD knowledge base and A/B/C questions (draft by D, for A to review)

Source: MedQuAD (Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to Question
Answering*, BMC Bioinformatics 2019), https://github.com/abachaa/MedQuAD, CC BY 4.0.
47,457 question-answer pairs from 12 NIH sites; 16,407 keep their answers (the three MedlinePlus
sets, including drug information, had answers removed for copyright).

Rebuild (deterministic, about 20 s; clones MedQuAD into `data/raw/MedQuAD` on first use):

    python -m evrag.data.medquad --per-scenario 60

| File | Content |
|------|---------|
| `corpus.jsonl` | **not committed** (20 MB); rebuild it. One chunk per line: `doc_id` = `<site>_<file>-<pid>#<chunk>`, `text` = `"<Focus> — <Section>\n\n<chunk body>"`, chunks of at most 150 words. 26,966 chunks after de-duplication, boilerplate removal and hold-outs. |
| `corpus_all.jsonl` | not committed; every chunk before hold-outs, so `scripts/d_run_medquad_trial.py` can embed them once and derive the retriever cache for any rebuild. |
| `dataset.csv` | 180 questions (A/B/C × 60; dev 45 / test 15 each) in the project schema plus audit columns. |
| `manifest.json` | MedQuAD commit, parameters, counts, and for every B/C question which pages and QA pairs were held out, with its final overlap values. |

`dataset.csv` columns: the shared ones (`question_id`, `question`, `gold_answers` as a JSON list,
`scenario`, `split`, `gold_doc_ids` as a JSON list, `source_doc_id`, `evidence_scenario_status`) and
`focus` (page name), `qtype` (MedQuAD question type), `source` (NIH site), `leak_overlap` (largest
share of the gold answer's word 8-grams found in one chunk outside the asked and summary sections)
and `sentence_overlap` (largest share of one gold sentence's content words found in such a chunk).

Scenarios:

| | Construction | Desired behaviour |
|---|---|---|
| A sufficient | the answer's chunks are in the corpus (`gold_doc_ids`) | answer |
| B missing | every page about the concept and its neighbours is held out | abstain |
| C distracting | the asked section and the summary sections (overview, points to remember) are held out across the neighbourhood; the page's other sections remain | abstain |

How the labels are made valid against the whole corpus:

- Concept = pages with the same name, or the same UMLS CUI set when it contains at least one
  specific CUI (on at most 40 pages). Neighbours = groups sharing a specific CUI, or whose name
  contains / is contained in this name (subtype and parent pages), one hop.
- Duplicate chunk bodies are merged within a concept only; a body found on more than 40
  concepts' pages (HPO table legends, video controls, site credits) is dropped as boilerplate.
- Every candidate, whatever its scenario, is rejected when a chunk outside its asked and summary
  sections copies at least 10% of the gold answer's 8-grams or holds at least 70% of the content
  words of one gold sentence (restatement). Rejected candidates are dropped, not recycled into
  another scenario, so A, B and C are drawn from the same population.
- Page titles that are booklet titles rather than condition names ("What I need to know about
  ...", "Nutrition for ...") are excluded; the templated questions they produce are garbled.

The labels are automatic (`evidence_scenario_status=auto`); the proposal requires a manual audit
of every B/C item before the labels are frozen. Known limits of this draft: most accepted
questions come from NINDS pages (other sites' overviews restate their sections and are rejected),
"treatment" and "outlook" dominate the question types, and 15 test items per scenario is too few
to separate methods — raise `--per-scenario` once the design is agreed.

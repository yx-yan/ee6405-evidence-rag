# MedQuAD knowledge base and A/B/C questions (draft by D, for A to review)

Source: MedQuAD (Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to Question
Answering*, BMC Bioinformatics 2019), https://github.com/abachaa/MedQuAD, CC BY 4.0.
47,457 question-answer pairs from 12 NIH sites; 16,407 keep their answers (the three MedlinePlus
sets, including drug information, had answers removed for copyright).

Rebuild (deterministic, about 20 s; clones MedQuAD into `data/raw/MedQuAD` on first use):

    python -m evrag.data.medquad --per-scenario 60

| File | Content |
|------|---------|
| `corpus.jsonl` | **not committed** (20 MB); rebuild it. One chunk per line: `doc_id` = `<site>_<qid>#<chunk>`, `text` = `"<Focus> — <Section>\n\n<chunk body>"`, chunks of at most 150 words. 25,890 chunks after de-duplication and hold-outs. |
| `dataset.csv` | 180 questions (A/B/C × 60; dev 45 / test 15 each) in the project schema plus audit columns. |
| `manifest.json` | MedQuAD commit, parameters, counts, and for every B/C question which pages and QA pairs were held out and its final leak overlap. |

`dataset.csv` columns: the shared ones (`question_id`, `question`, `gold_answers` as a JSON list,
`scenario`, `split`, `gold_doc_ids` as a JSON list, `source_doc_id`, `evidence_scenario_status`) and
`focus` (page name), `qtype` (MedQuAD question type), `source` (NIH site), `leak_overlap` (largest
share of the gold answer's word 8-grams found in any remaining chunk; 0 for A).

Scenarios:

| | Construction | Desired behaviour |
|---|---|---|
| A sufficient | the answer's chunks are in the corpus (`gold_doc_ids`) | answer |
| B missing | every page about the concept and its neighbours is held out | abstain |
| C distracting | only the asked section is held out across the neighbourhood; the page's other sections (usually the overview) remain | abstain |

Concept = pages with the same focus name or the same UMLS CUI set. Neighbours = groups sharing a
CUI listed on at most 40 pages (one hop). Duplicate chunk bodies are merged before labelling, and
B/C candidates are rejected when a remaining chunk shares at least 10% of the gold answer's
8-grams. The labels are automatic (`evidence_scenario_status=auto`); the proposal requires a
manual audit of every B/C item before the labels are frozen.

Questions come from the common-condition sites (CancerGov, NIDDK, NINDS, SeniorHealth, NHLBI,
CDC) and the types symptoms, treatment, causes, exams and tests, outlook, prevention,
susceptibility, complications, considerations and stages; GARD, GHR and MedlinePlus Health
Topics pages are corpus-only.

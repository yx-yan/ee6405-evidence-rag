# TREC 2017 LiveQA-Med: real consumer questions over the MedQuAD corpus (draft by D)

Source: TREC 2017 LiveQA Medical Task test set (Ben Abacha et al., NLM/NIST),
https://github.com/abachaa/LiveQA_MedicalTask_TREC2017, plus the 2,479 NLM-judged MedQuAD answers
shipped with MedQuAD (https://github.com/abachaa/MedQuAD, `QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip`).
Research test data released by NLM; cite the TREC 2017 LiveQA overview and MedQuAD (BMC Bioinformatics 2019).

Rebuild: put `test.xml` (the "w-summaries" file), the qrels `.txt` and the answers `.csv` under
`data/raw/liveqa_med/`, then

    python -m evrag.data.liveqa

| File | Content |
|------|---------|
| `dataset.csv` | 104 questions in the project schema. `question` = the public's subject + message as written; `summary` = NLM's one-line rewording; `focus` / `qtype` = NLM annotations; `gold_answers` = NLM reference answers (JSON list). |
| `manifest.json` | for every question the NLM-judged MedQuAD answers and which of them are in our corpus. |

Scenario is `R` (real question; nothing is held out). `evidence_scenario_status`:

- `judged` (36): at least one MedQuAD answer that NLM rated 3-Incomplete or 4-Excellent for this
  question is in our corpus; `gold_doc_ids` lists its chunks (retrieval gold for real questions).
- `reference_only` (68): the judged answers come from the MedlinePlus sets whose answers MedQuAD
  removed (encyclopedia, drugs, herbs), so only the reference answers are available for grading.

Split: dev 52 / test 52, stratified by status. Of the 365 judged-good answers, 91 are in the corpus,
241 are outside it, 16 point at MedQuAD pairs we do not have and 17 did not match the corpus text.

Use: run the methods with `scripts/d_run_medquad_trial.py --dataset data/liveqa/dataset.csv`; grade
answers against `gold_answers` (LLM or human) and, for `judged` rows, measure whether a gold chunk was
retrieved and used.

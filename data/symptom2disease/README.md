# Symptom2Disease: colloquial symptom descriptions over the MedQuAD corpus (draft by D)

Source: Kaggle "Symptom2Disease" (niyarrbarman): 1,200 one-paragraph symptom descriptions written in
the first person, labelled with one of 24 diseases (50 per label; 1,153 distinct texts). The license
on the Kaggle page still has to be confirmed; the file is not committed (`data/raw/symptom2disease/`).

Rebuild (defaults: 10 descriptions per label, dev 5 / test 5 each):

    python -m evrag.data.symptom2disease

| File | Content |
|------|---------|
| `dataset.csv` | 240 questions (scenario `S`). `question` = the description; `gold_answers` = `[label]`; `focus` = the MedQuAD page(s) for the label; `gold_doc_ids` = those pages' Symptoms chunks (or all their chunks). |
| `manifest.json` | the label -> page mapping actually found in the corpus and counts per label. |

23 of the 24 labels have a MedQuAD page (`evidence_scenario_status=mapped`); "Cervical spondylosis"
has none (`no_page`), so retrieval metrics skip it while answer grading can keep it.

Use: this is the symptom-description test for the open-question mode. Metrics: whether a chunk of the
label's page is in the top k (retrieval), whether the label (or its page name) appears among the
conditions the answer names (coverage), and abstention rate. A closed-book run is needed as a control,
because small local models already name the right label about half the time without evidence
(`docs/llm_selection.md`).

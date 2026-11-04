# MedQuAD trial for the evidence-aware RAG project (draft by D, 2026-11-04)

**Status: proposal for the team, not a decision.** Data is A's area; this branch (`d-medquad-trial`)
changes nothing under `data/processed/` and is not merged. Everything here can be rebuilt with two
commands (see `data/medquad/README.md`).

## Why

Our smoke runs on MTS-Dialog showed that the questions cannot be answered by retrieval: they say
"the patient" while the corpus holds notes from 1,201 unrelated patients, so the question's own note
is in the top 5 only 33% of the time (48% with the scenario text), gold answers use facts that are
only in the dialogue, and there are no A/B/C evidence labels. The chatbot we want answers
consumer-health questions ("what could this be, what can I take") from a medical knowledge base.

## What MedQuAD gives us

MedQuAD (Ben Abacha & Demner-Fushman 2019, CC BY 4.0): 47,457 QA pairs from 12 NIH sites, 16,407
with answers (MedlinePlus drug/encyclopedia answers were removed for copyright). Each page is one
condition with sections (overview, symptoms, causes, treatment, outlook, ...). Questions name the
condition, so retrieval is well-posed, and sections can be held out to construct the proposal's
three evidence scenarios on one fixed corpus.

## The draft dataset (`python -m evrag.data.medquad`)

| | |
|---|---|
| Knowledge base | 26,966 chunks (<= 150 words, header `"<Focus> — <Section>"`) from 16,407 answers after merging duplicates within a concept and dropping 2,526 boilerplate chunks |
| Questions | 180 = A/B/C x 60, dev 45 / test 15 each; one question per concept; sources CancerGov, NIDDK, NINDS, SeniorHealth, NHLBI, CDC |
| A sufficient | gold chunks in the corpus -> answer |
| B missing | whole concept + neighbours (shared specific CUI, subtype/parent by name) held out -> abstain |
| C distracting | asked section + summary sections held out; other sections of the page remain -> abstain |
| Validity gates | a candidate (any scenario) is rejected if a remaining chunk copies >= 10% of its gold 8-grams or restates >= 70% of one gold sentence's content words; rejected candidates are dropped, not recycled; booklet-style page titles excluded |

Labels are automatic (`evidence_scenario_status=auto`). The proposal requires a manual audit before
freezing; section "Label audit" below reports a first pass.

## Retrieval check (no LLM, dev, 45 per scenario, MiniLM, top 5)

| Scenario | Same-page chunks in top 5 | Gold chunk in top 5 | Gold at rank 1 | Top-1 cosine (median) |
|---|---|---|---|---|
| A | 45/45 | 41/45 (91%) | 28/45 | 0.75 |
| B | 0/45 (held out) | - | - | 0.60 |
| C | 44/45 (mean 2.6 chunks) | - | - | 0.69 |

Retrieval finds the right page when it exists (A), returns nothing about the condition when the
page is gone (B), and returns the page's other sections when only the asked section is gone (C).
The score gap between A and B (0.75 vs 0.60) is what C's threshold method can exploit; A and C
overlap, which is exactly why a similarity threshold alone cannot handle scenario C.

## Live trial (gpt-4.1-mini, dev, 20 per scenario, baseline vs evidence)

Files: `results/runs/{baseline,evidence}_medquad_dev60.jsonl`, `results/runs/medquad_trial_dev60_summary.json`.
141 LLM calls, 115k prompt + 6k completion tokens, about $0.06. Judge prompt `evidence_judge_v2`,
0 parse failures.

| Method | A: answered / abstained | B: answered / abstained | C: answered / abstained | Mean latency (A / B / C) |
|---|---|---|---|---|
| baseline (always answers) | 20 / 0 | 20 / 0 | 20 / 0 | 1.6 s / 1.4 s / 1.3 s |
| evidence (judge, then answer) | 17 / 3 | 2 / 18 | 2 / 18 | 2.2 s / 1.0 s / 1.1 s |

The evidence method abstains on 36 of the 40 questions whose answer is not in the corpus and
keeps 17 of the 20 answerable ones (gold chunk was in the top 5 for 19 of them). The baseline
answers all 40 unanswerable questions. Answer correctness and grounding are graded in the next
section.

The seven "wrong" evidence decisions, read one by one:

| Case | What happened | Whose problem |
|---|---|---|
| B, treatments for Neurosyphilis | another page ("Neurological Complications of AIDS — Treatment") says "penicillin to treat neurosyphilis"; the judge used it and the answer matches the gold | label leak: the hold-out missed a mention on a related page |
| C, outlook for Troyer Syndrome | the remaining Symptoms section says "Affected people typically have a normal life expectancy"; answer matches the gold | label leak: prognosis stated outside the Outlook section (restatement score 0.60, below the 0.70 gate) |
| C, symptoms of Peyronie's Disease | the judge accepted the Complications section as symptoms; the answer lists complications, not the gold's lumps/pain/curvature | judge error: a true "distracting" failure, the case the project is about |
| B, treatments for Knee Replacement | "Knee Replacement" is a procedure page, not a condition; the judge answered from arthritis treatment pages | question quality: title should be excluded |
| A, treatments for Trichinellosis | gold chunk at rank 1 says "safe and effective prescription drugs are available"; judge wanted drug names and said INSUFFICIENT | judge too strict: prompt tuning target |
| A, "what else can be done to prevent" Marine Toxins | gold chunk at rank 1; judge wanted "additional" measures | judge too strict on an oddly templated CDC question |
| A, causes of Age-related Macular Degeneration | the Causes chunk was not in the top 5 (the only A retrieval miss); judge abstained on overviews | retrieval miss; abstaining was right given the evidence |

So of the four answered B/C questions, two are label errors the audit must catch, one is a
genuine judge failure, one is a bad question; of the three A abstentions, two are prompt-tuning
targets and one is a retrieval miss.

## Label audit (first pass)

_To be filled: per-scenario verdicts for the 60 trial questions._

## How this was checked

Three independent code reviews (label validity, integration with B's retriever/contract,
sampling and determinism) found: umbrella CUIs ("Syndrome") merging unrelated pages, cross-concept
duplicate chunks surviving hold-outs under the held page's name, A questions being the
leak-rejected complement of B/C, subtype/parent pages escaping hold-outs, summary sections
restating held-out answers, and malformed questions from NIDDK booklet titles. All are fixed in
the current build and covered by `tests/test_member_d_medquad.py` (54 tests pass).

## Known limits and next steps

- Most accepted questions are NINDS pages (clean overview/section structure); NIDDK, NHLBI and CDC
  pages are often rejected because their overviews restate the sections. Per-source quotas would
  need relaxed gates or more pages.
- "treatment" and "outlook" dominate the question types; "symptoms" rarely survives the gates.
- 15 test questions per scenario cannot separate methods; raise `--per-scenario` once agreed.
- Real consumer questions: TREC 2017 LiveQA-Med (104 questions with reference answers; 2,479
  MedQuAD answers judged for them) can serve as a realistic test set.
- faiss-cpu 1.15.1 dies with SIGILL on this server at this index size; the trial used B's exact
  numpy backend (`--backend numpy`), which ranks identically.

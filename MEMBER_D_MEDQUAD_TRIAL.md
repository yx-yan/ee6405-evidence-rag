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

### Answer grading (preliminary, `results/d_medquad_trial/grading_dev60.json`)

Every answer was graded against the gold answer and the five retrieved chunks by Claude agents
(one grader per 15 answers; strict rules in the file's note). This is a first pass to show what
the comparison looks like, not E's metric.

| Method / scenario | correct | partially correct | incorrect | prose refusal | formal abstention |
|---|---|---|---|---|---|
| baseline / A (ideal: answer) | 19 | 1 | 0 | 0 | 0 |
| evidence / A | 17 | 0 | 0 | 0 | 3 |
| baseline / B (ideal: abstain) | 1* | 2 | 4 | 13 | 0 |
| evidence / B | 1* | 0 | 1 | 0 | 18 |
| baseline / C (ideal: abstain) | 3* | 5 | 5 | 7 | 0 |
| evidence / C | 1* | 0 | 1 | 0 | 18 |

\* "correct" on B/C means the label leaked: the corpus did contain the answer somewhere (see the
label audit), so these count against the data, not the method.

Reading: on the 40 questions whose answer is not in the corpus, the baseline produced 16
incorrect or misleading answers (9 incorrect, 7 partially correct, 6 of them drawn from a
chunk about a different condition) and refused in prose on 20; the evidence method produced 2
incorrect answers and abstained formally on 36. On the 20 answerable questions the baseline
answered all 20 (19 correct) and the evidence method answered 17, all correct. Two consequences
for the team: (1) gpt-4.1-mini with B's prompt refuses in prose half the time when the evidence
is missing, so E's scorer must detect prose refusals or the baseline is scored as if it had
hallucinated; (2) the evidence judge's cost is coverage (3/20 over-refusals), which is where
prompt tuning on dev should go.

## Label audit (first pass, `results/d_medquad_trial/label_audit_dev60.json`)

Each of the 60 trial questions was checked by a Claude agent that read the question's top-10
retrieved chunks, every remaining chunk of the same page, the held-out page list, and grepped the
corpus for synonyms, parent conditions and the gold answer's facts. This stands in for, and does
not replace, the manual audit the proposal requires.

| Scenario | valid | partial (a related page covers part of the answer) | leaked (a remaining chunk states the answer) | unclear |
|---|---|---|---|---|
| A (20) | 19 | - | - | 1 (the AMD "causes" gold is a fragment about drusen) |
| B (20) | 11 | 8 (6 of them outside the top 5) | 1 (Neurosyphilis: a related page says "penicillin to treat neurosyphilis") | - |
| C (20) | 6 | 8 (6 outside the top 5) | 6 (4 inside the top 5; e.g. the Symptoms section of Troyer syndrome states the prognosis; Arrhythmia—Treatment names the drugs for atrial fibrillation) | - |

B labels are mostly sound: when a related page covers part of the answer it is usually a parent
or subtype page ranked 6-10, which the methods never see. C labels are the weak point: NIH pages
restate prognosis inside Symptoms or Treatment sections, which the summary hold-out and the
restatement gate (70% of a sentence's content words) do not catch when the wording differs.
Either C holds out every section of the page except the overview (making C a "page summary
only" scenario), or C candidates are audited by hand before use; the automatic gates alone are not
enough for C.

Note that the evidence judge still abstained on 18 of the 20 C questions, including 3 of the 4
whose answer had leaked into the top 5: the judge errs on the conservative side there.

## How this was checked

Three independent code reviews (label validity, integration with B's retriever/contract,
sampling and determinism) found: umbrella CUIs ("Syndrome") merging unrelated pages, cross-concept
duplicate chunks surviving hold-outs under the held page's name, A questions being the
leak-rejected complement of B/C, subtype/parent pages escaping hold-outs, summary sections
restating held-out answers, and malformed questions from NIDDK booklet titles. All were
addressed and are covered by `tests/test_member_d_medquad.py` (54 tests pass). A second,
independent pass then re-checked each finding against the final build
(`results/d_medquad_trial/review_verification.json`): 3 fixed outright (cross-concept
duplicates, A sampling bias, summary sections as question type), 4 fixed in mechanism with a
residual (umbrella CUIs, the leak gate's bullet lists, the title blocklist, name-based
neighbourhoods missing plural/compound forms such as Leukodystrophy/Adrenoleukodystrophy), and
2 still reproducible on a few items: C sections restating the answer in other words and B family
pages under another name. These two are the reason the label audit above is required before the
labels are frozen.

## Known limits and next steps

- Scenario C needs a stricter construction or a manual audit (see the label audit); B needs the
  parent/subtype hold-out widened by name (e.g. "Epilepsy" for a seizure-disorder subtype).
  Independent re-verification of the review findings on the final build confirms both residual
  risks: the restatement gate has no stemming or synonyms, so paraphrases below 70% content-word
  overlap pass, it splits only on sentence punctuation (period-free " - " bullet lists are one
  long unit), and family pages under another name are not reached. A should audit every B/C
  item by reading the top-5 (ideally top-10) chunks whose page is not in `manifest.json`
  `held_pages`.
- Page titles that are procedures or topics ("Knee Replacement") still slip through the title
  filter; a short stop-list of procedure words would remove them.
- Most accepted questions are NINDS pages (clean overview/section structure); NIDDK, NHLBI and CDC
  pages are often rejected because their overviews restate the sections. Per-source quotas would
  need relaxed gates or more pages.
- "treatment" and "outlook" dominate the question types; "symptoms" rarely survives the gates.
- 15 test questions per scenario cannot separate methods; raise `--per-scenario` once agreed.
- Real consumer questions: TREC 2017 LiveQA-Med (104 questions with reference answers; 2,479
  MedQuAD answers judged for them) can serve as a realistic test set.
- faiss-cpu 1.15.1 dies with SIGILL on this server at this index size; the trial used B's exact
  numpy backend (`--backend numpy`), which ranks identically.

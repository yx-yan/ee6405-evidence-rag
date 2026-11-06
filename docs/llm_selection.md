# Choosing the shared LLM

**Recommendation: `qwen3:4b-instruct-2507-q4_K_M`, run locally through Ollama.** It passes every gate, gives the best judge and near-best answers, and works with `llm.py` as it is. The team still needs to approve this before `configs/default.yaml` changes.

**Scope:** the model comparison below ran on MTS-Dialog, which the project no longer uses as an evaluation dataset. Qwen was then checked on MedQuAD ([MedQuAD check](#medquad-check)) and all three current datasets were probed for prior knowledge ([Prior knowledge](#prior-knowledge-of-the-three-datasets)). Only Qwen ran on the MedQuAD trial, so the ranking of models is still an MTS-Dialog result.

**We found no evidence that the local models memorised MTS-Dialog.** This does not cover large API models, because `gpt-4.1-mini` was not tested. Picking a small model is therefore not a fix for memorisation, and the report should not claim it is. The reasons to use it are that it is free, pinned to one version and reproducible.

**The MedQuAD check supports the choice but does not finish it.** Qwen's answer-or-abstain decisions match the earlier `gpt-4.1-mini` trial closely, but its answers have not been graded.

## MedQuAD check

This used the first 20 dev questions per scenario, `bge-base-en-v1.5`, top 5, the `evidence_judge_v2` prompt, the numpy backend and a 120 s timeout. There were 139 calls and no parse failures. Raw outputs are in `results/llm_medquad/`.

| `evidence` method | Earlier trial: `gpt-4.1-mini` + MiniLM | This run: Qwen + bge-base |
|---|---|---|
| A: answered / abstained (should answer) | 17 / 3 | 17 / 3 |
| B: answered / abstained (should abstain) | 2 / 18 | 1 / 19 |
| C: answered / abstained (should abstain) | 2 / 18 | 1 / 19 |
| p95 latency, A / B / C | 3.3 / 2.0 / 2.2 s | 9.5 / 6.0 / 4.0 s |

Both the LLM and the embedding model changed, and each scenario has 20 questions, so this shows the setup works. It does not show that Qwen is better.

- **The three A abstentions are the same questions in both runs.** Two had the gold chunk at rank 1, so the judge was too strict. The third (AMD causes) was a retrieval miss.
- **Two B/C questions were answered.** Troyer syndrome (C) is a known label leak. Pericarditis (B) was audited as valid, so it may be a real judge failure and needs a human read.
- **The baseline refuses in prose more often.** A regex estimate finds about 31 refusals on the 40 B/C questions, against about 21 for `gpt-4.1-mini` (manual grading of that run found 20). E's scorer has to detect these.
- **bge-base changes scenario C.** The dev A gold chunk is in the top 5 for 44 of 45 questions (MiniLM: 41), but only 39 of 45 C questions now retrieve a chunk from the same page (MiniLM: 44). The other 6 behave more like scenario B.
- **Scores shift upward.** Median top-1 cosine is 0.83 / 0.73 / 0.77 for A / B / C (MiniLM: 0.75 / 0.60 / 0.69), so any threshold tuned on MiniLM needs retuning.

## Prior knowledge of the three datasets

Each model answered without any evidence. MedQuAD used 135 dev questions, LiveQA-Med 104 questions and Symptom2Disease 240 descriptions. Raw replies are in `results/dataset_knowledge/`.

| Model | MedQuAD word recall | LiveQA word recall | Symptom2Disease accuracy (95% CI) | Verbatim continuation |
|---|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | 0.214 | 0.124 | 50.8% (44.6–57.1) | 0% |
| `phi4-mini:3.8b` | 0.169 | 0.087 | 43.3% (37.1–50.0) | 0% |
| `llama3.2:3b` (incomplete) | 0.176 | not valid | not run | not run |

- **No model reproduced dataset text.** Continuation shared no 8-word run with the true text on any dataset.
- **Symptom2Disease is answerable without retrieval about half the time.** Chance is about 4% with 24 labels. A correct answer on this dataset therefore cannot be credited to evidence on its own, so results need a closed-book comparison.
- **Word recall is not correctness.** The `grade` step was not run, so the MedQuAD and LiveQA columns only measure word overlap. MedQuAD recall is similar across A, B and C (0.20 to 0.22 for Qwen).
- **`llama3.2:3b` stopped early.** It has 44 of 104 LiveQA items and no classify or continuation results, and it refused 68% of those LiveQA items.

## Results on MTS-Dialog

All runs used the dev split, 2026-11-04 to 05, on an Apple M5 with 16 GB, at temperature 0 with `max_tokens` 256. There were 1,360 calls and none failed in the configurations below. Raw replies are in `results/llm_screen/`.

| Model | P1 gap | P1 refusals | P2 verbatim | P3 follows note | P4a F1 | P4b balanced acc. (95% CI) | Over-refusal | p95 latency |
|---|---|---|---|---|---|---|---|---|
| **`qwen3:4b-instruct-2507-q4_K_M`** | 0.112 | 1.7% | 0% | 87.5% | 0.543 | **0.833** (0.75–0.92) | **33.3%** | 4.3 s |
| `phi4-mini:3.8b` | 0.055 | 1.7% | 0% | **95%** | 0.466 | 0.767 (0.68–0.85) | 46.7% | 3.5 s |
| `llama3.2:3b` | not valid | 100% | 0% | 90% | 0.476 | 0.750 (0.65–0.85) | 43.3% | 3.0 s |
| `gemma4:e4b`, thinking off * | 0.066 | 1.7% | not valid | 82.5% | 0.537 | 0.817 (0.73–0.90) | 36.7% | 5.1 s |
| `gemma4:12b`, thinking off * | not valid | 75% | not valid | 87.5% | **0.549** | 0.650 (0.57–0.73) | 70.0% | 16.4 s |
| `llama2:7b` (control only) | 0.040 | 6.7% | 0% | 95% | – | – | – | 3.4 s |

\* Supplementary runs, outside the original decision rule. See [Gemma 4](#gemma-4).

What each probe does:

| Probe | Items | Test |
|---|---|---|
| P1 closed book | 60 | Question only. Gap = recall of gold-answer words, minus recall of a different note's gold answer from the same section. |
| P2 continuation | 40 | Raw completion of a note's first half. Verbatim = shares an 8-word run with the true second half. |
| P3 counterfactual | 40 | A drug or condition in the note is swapped. Does the answer use the new term? |
| P4a answering | 60 | Source note plus four distractors. Token F1 against the gold answer. |
| P4b judging | 30 + 30 | SUFFICIENT with the source note present; INSUFFICIENT when no retrieved note can answer. |

## Is there memorisation?

- **No verbatim recall.** All four models tested on P2 scored 0% on both the true text and the control.
- **The P1 gap looks like medical common sense.** We read Qwen's six highest-scoring replies. They were plausible guesses, not recall: it named penicillin allergies for patients with none, and listed common drugs in place of the real ones.
- **P3 shows the models read the note.** The original term appeared in at most 2.5% of answers. All five of Qwen's misses left the term out or said the evidence lacked it. None went back to the original term.

The caveat is that the P1 reading rests on six replies from one model. If the team treats the P1 gap as memorisation instead, `phi4-mini:3.8b` would rank ahead of Qwen on the memorisation criterion. These probes can also only detect memorisation that shows up in outputs, so they cannot prove a model never saw the data. MTS-Dialog notes come from the public MTSamples site, so no model here, including Llama 2, can be assumed to be clean.

## How the decision was made

The rules were fixed before any model ran.

1. **Gates:** parse failures at most 5%, P4b balanced accuracy at least 0.70, p95 under 30 s, and able to finish on a 16 GB Mac. Qwen, Phi and Llama 3.2 pass.
2. **Rule for no memorisation signal:** since P2 shows none and P1 looks like common sense, we chose on P4 and reproducibility instead of lowest memorisation score. The rule was written to include the `gpt-4.1-mini` reference, which is missing, so this conclusion covers local models only.
3. **Choice:** Qwen is highest on both P4a and P4b. The confidence intervals overlap with Phi's, so this is the best available estimate and not a proven difference.

## Gemma 4

Gemma 4 thinks by default. With our `max_tokens` of 256, the thinking used up the budget, and 39 of 40 chat replies in the smoke test across the two models had no usable answer (19 of 20 for `e4b`, all 20 for `12b`). It therefore fails in the default setup.

Turning thinking off works, using `reasoning_effort: "none"`, so we tested that as a supplement:

- **`gemma4:e4b`** is a close second to Qwen. Using it would need a change to `llm.py`, and Qwen needs none.
- **`gemma4:12b`** fails the judge gate. It also refused 75% of P1, used about 16 GB of memory and pushed swap to about 11 GB.
- **P2 is not valid for either model**, because raw completion produced repeated fragments.

## What to expect from Qwen on MTS-Dialog

- **Over-refusal is the main weakness.** The judge rejected 33% of questions it should have accepted. In 7 of Qwen's 10 false refusals, it said the excerpts describe different patients or no single excerpt covers everything. This comes from rules 2 and 3 of the judge prompt, so D can tune `evidence_judge_v1` on dev.
- **The judge was strict on negatives.** All 30 were correctly rejected.
- **Answers blend patients.** In a 20-item sample of P4a, about 9 answers took details from a distractor note. One gave the 85-year-old patient's hearing problems as the reason an exam was not done. This happens even with the right note present, and it is the kind of failure our methods need to report.
- **`baseline` and `evidence` both run end to end** on 5 dev questions with no code changes. One call took 19.6 s while the model loaded, so the shared timeout is now 120 s.

## Sample review

We read the samples before running any model and changed them where needed:

- 32 P4b negatives were removed because another patient's note could answer them. Examples are "What are the patient's reported social and substance use histories?" and questions where a synonym got past the filter ("double vision" and "diplopia").
- 1 P3 item was removed because the swap contradicted the question.
- Negatives now need a rare word from the question that no evidence note contains. Positives need at least 0.8 gold-word support. Distractors are de-duplicated.

Exclusions are listed in `scripts/llm_screen.py`. The samples were frozen before the first full run (`samples.json`, SHA-256 starting `8baba230`).

**This review, and the checks of P3, P4a and P1 replies above, were done by Claude, not by a team member.** Someone on the team should spot-check them before we cite them. This also means the "about 9 of 20" figure has not been human-verified.

## Limitations

- Samples are small, and most confidence intervals overlap.
- Oracle evidence is a best case. It says nothing about end-to-end accuracy with real retrieval.
- Only 167 of 953 dev questions have at least 0.7 gold-word support in their source note, so these samples favour questions the notes can answer.
- `llama3.2:3b` and `gemma4:12b` mostly refused P1, so their low scores say nothing about memory.
- Token F1 penalises correct answers worded differently from the gold answer.

## Open issues for the team

1. **Approve the LLM.** `llm.model` in `configs/default.yaml` is still `TODO`. Set it once the team accepts Qwen and someone has read its 19 `evidence` answers, starting with Pericarditis.
2. **Audit the MedQuAD labels.** The first-pass audit found only 6 of 20 C items and 11 of 20 B items fully valid. It was done by Claude, so a person needs to audit B and C before the data is frozen.
3. **Connect LiveQA-Med and Symptom2Disease.** They exist only as local files under `data/raw/`, which is not committed. Neither has a builder, and `data.dataset` holds one dataset.
4. **Large-model reference.** Running the closed-book probes on `gpt-4.1-mini` is the only direct test of memorisation in a large API model.
5. **Out-of-date docs.** The README, proposal and ROLES still describe SQuAD 2.0.

## Reproducing

```bash
brew install ollama
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
ollama pull qwen3:4b-instruct-2507-q4_K_M

# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
LLM_API_KEY=ollama        # any non-empty value

PYTHONPATH=src python scripts/llm_screen.py run --model qwen3:4b-instruct-2507-q4_K_M
PYTHONPATH=src python scripts/llm_screen.py summarize

# MedQuAD check; writes to results/runs/
PYTHONPATH=src python scripts/d_run_medquad_trial.py --split dev --per-scenario 20 --methods baseline evidence --backend numpy

# Prior knowledge; needs data/raw/liveqa_med/test.xml and data/raw/symptom2disease/Symptom2Disease.csv
PYTHONPATH=src python scripts/dataset_knowledge.py build
PYTHONPATH=src python scripts/dataset_knowledge.py run --model qwen3:4b-instruct-2507-q4_K_M
PYTHONPATH=src python scripts/dataset_knowledge.py summarize
```

Model digests and the Ollama version are in `results/llm_screen/models.json`.

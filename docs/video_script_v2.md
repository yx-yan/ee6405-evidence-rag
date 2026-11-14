# Video script v2 (8:13)

Second version of [video_script.md](video_script.md). What changed:

- **Theory slides.** Each member's part now opens with one theory slide (formulas plus a one-line
  intuition), as the brief asks for "slides for theory". The narration explains the idea behind
  each formula; it does not derive it.
- **Shorter demo.** Part 1 goes from 4:15 to 2:45. The out-of-domain example is folded into the
  missing-evidence one, and the MVO graph example moves to C's part.
- **Plainer language.** Fewer numbers per sentence; every part ends on one takeaway.

**Part 1 (0:00–2:45)** is the demo, narrated by E. **Part 2 (2:45–7:48)** has each member explain
their part: A → B → C → D → E. **Wrap-up (7:48–8:13)** by E.

**Timing.** About 930 words of narration in 8:13, roughly 115 words a minute: room for demo
clicks, for viewers to read a slide, and for natural pauses. The limit is 10:00, so 1:47 is left
free; use it only if a part needs it after recording.

## Before recording

Same as v1: run `python scripts/video_figures.py` and `streamlit run app/app.py`, run every demo
example once so the models are warm, and read at a relaxed pace. References are shown as
a caption while their part plays and listed again on the end card; they are not read aloud.

## Theory slides

Seven new slides, 1920x1080, in the shared figure style (`src/evrag/figures.py`), with the
top-right camera corner (right 26%, top 40%) left empty. Each has a title, one to three formulas
(matplotlib mathtext is enough) and a one-line plain-English intuition under each formula. The
symbols and numbers below match the code; do not change them.

| Slide | Content | Code it matches |
|---|---|---|
| `theory_task` | Answer or abstain as one decision: given question $q$ and retrieved passages $D$, output an answer $y$ or ⊥ ("not enough evidence"). Reliability = (correct answers + correct refusals) / all questions. Three test scenarios: answerable, missing evidence, out of domain. | `evaluation/metrics.py` |
| `theory_explain` | Occlusion: $a_i = \log p(y \mid x) - \log p(y \mid x_{\setminus i})$, how much the answer's probability drops when word $i$ is masked. LIME fits a linear model on random masks; SHAP averages each word's contribution over all mask combinations (Shapley values). All three score the same answer $y$ by teacher forcing. | `explain.py` |
| `theory_chunking` | Sliding window: passages of 150 words, a new one every 120 words (overlap 30). Any answer of 30 words or fewer always fits whole in one passage. Unanswerable questions: *missing evidence* = delete the gold passages from the corpus; *out of domain* = take questions from the other two sets. | `data/disease_sets.py` `chunk_spans` |
| `theory_dense` | Bi-encoder: $\mathbf{q} = E(q)$, $\mathbf{d} = E(d)$ (MiniLM, mean pooling). $\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}\cdot\mathbf{d}}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{d}\rVert}$. Vectors are normalised, so FAISS inner product = cosine. Passages are encoded once; a question costs one encoding. | `retriever.py` |
| `theory_sparse_graph` | TF-IDF: $\mathrm{tf}(t,d)\cdot\mathrm{idf}(t)$. BM25: $\sum_t \mathrm{idf}(t)\,\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf} + k_1(1 - b + b\,\frac{|d|}{\mathrm{avgdl}})}$ with $k_1 = 1.5$, $b = 0.75$: repeated words saturate, long passages are normalised. Graph: passages ↔ words; direct score = share of the question's IDF weight a passage contains; one hop adds a neighbour with score × 0.35 / $\log_2(\mathrm{df}+1)$, so rare words spread and common words do not. | `evaluation/retrieval_eval.py` `BM25`, `methods/graph_rag.py` |
| `theory_bart` | Encoder–decoder transformer: the encoder reads `question: … context: [1] … [2] …`, the decoder writes the answer while attending to it (cross-attention). Pretrained as a denoiser, fine-tuned with $\mathcal{L} = -\sum_t \log p(y_t \mid y_{<t}, x)$. Decoding: beam search, 4 beams. Training: 66k examples, 3 epochs, lr 5e-5, batch 16. | `generator.py`, `docs/generator.md` |
| `theory_abstain` | Two signals: $s_1$ = top-1 cosine score; $s_2 = \mathrm{coverage} = \frac{\sum_{w \in q \cap d} \mathrm{idf}(w)}{\sum_{w \in q} \mathrm{idf}(w)}$. Rule: answer if $s_1 \ge \tau_1$ and $s_2 \ge \tau_2$; else expand the query once (Rocchio: $q' = q + $ the 5 top TF-IDF words of the top 3 passages), retrieve again, check again; still failing → abstain. Thresholds on dev only, by Youden's $J$ = abstention recall − over-refusal rate. | `methods/evidence_rag.py`, `evaluation/tune.py` |

---

## Part 1 — Demo (E)

### 0:00–0:20 · The problem

**Show:** demo start page. Title overlay for the first 5 s: group name, the five names, "EE6405
Natural Language Processing".

> Hi, we're Group 9, and I'm Yan Yuxuan. If you ask a medical chatbot something it has no
> information about, it usually answers anyway, and sounds confident. In medicine that's
> dangerous. Our project teaches a chatbot when to say "I don't know".

### 0:20–0:50 · The task and the five methods

**Show:** `theory_task` (0:20–0:32), then `results/figures/video/pipeline.png`.

> We treat this as one decision: answer, or abstain. A good system answers the questions it can,
> and refuses the ones it can't. We call the share it gets right "reliability".
> We compare five methods. All use the same retriever and the same small generator, a BART model
> we fine-tuned ourselves. Three always answer. The last two, threshold_rag and our evidence_rag,
> are allowed to say no.

*References:* Lewis et al., Retrieval-Augmented Generation (NeurIPS 2020).

### 0:50–1:18 · It answers

**Show:** demo → **Answerable**, dataset *cancer*, method *evidence_rag* → **Generate**. Zoom on the
two score boxes, then on the answer.

> A normal question: "What are the symptoms of gallbladder cancer?" evidence_rag checks two
> things: how well the best passage matches, and how much of the question it covers. Both clear
> their bars, so it answers, and it lists the symptoms. Below, you can see exactly which passages
> it used.

### 1:18–1:43 · It looks again

**Show:** **Retrieve again** → **Generate**. Zoom on the grey box with the expanded query.

> "Who is at risk for testicular cancer?" The first search is just under the bar. Instead of
> giving up, it borrows the most telling words from the passages it found, adds them to the
> question, and searches again. This time it passes, and it answers.

### 1:43–2:23 · It says no

**Show:** **Missing evidence**: method **basic_rag** → **Generate**; then **evidence_rag** →
**Generate**. Then **Out of domain** → **Generate** (last 10 s).

> Here we deleted the right passages on purpose. "What is prostate cancer?" basic_rag answers
> anyway, with something about Gleason scores. It sounds medical, but it's not an answer.
> evidence_rag tries twice, the evidence stays too weak, and it says "not enough evidence".
> The same happens when we ask the cancer database about COVID: it refuses instead of making
> something up.

### 2:23–2:45 · Why did it say that?

**Show:** **Distractor** → scroll to the heatmap (2:23–2:33), then `theory_explain`.

> Every answer can be explained. Red words pushed the answer, so you can see which passage it
> really came from. We hide words one at a time and measure how much the answer's probability
> drops, and compare that with LIME and SHAP. Now, the team.

*References:* Lundberg & Lee, SHAP (NeurIPS 2017); Ribeiro et al., LIME (KDD 2016).

---

## Part 2 — Who did what

### 2:45–3:30 · A: the data

**Show:** `theory_chunking` (2:45–3:05), then `results/figures/video/datasets.png`, then code
`src/evrag/data/disease_sets.py` lines 56–61 (`chunk_spans`).

> Hi, I'm Nguyen Khanh Nam, and I built the data. We used three diseases from three sources, so
> our results don't hang on one dataset: coronavirus research papers from COVID-QA, and cancer and
> neurological pages from MedQuAD.
> Long texts are cut into 150-word passages that overlap by 30 words, so a short answer is never
> split in half.
> Then the key part: almost half the test questions should not be answered. For some we removed
> the right passages; others ask about a different disease.

*References:* Möller et al., COVID-QA (2020); Ben Abacha & Demner-Fushman, MedQuAD (BMC
Bioinformatics 2019).

### 3:30–4:18 · B: dense retrieval and basic RAG

**Show:** `theory_dense` (3:30–3:50), then code `src/evrag/retriever.py` lines 114–140 (`search`),
then `results/figures/video/basic_rag_errors.png`.

> Hi, I'm Xu Yiwen, and I built the retriever and basic RAG. A small model, MiniLM, turns every
> passage into a vector. A question becomes a vector too, and the passages pointing in the most
> similar direction, by cosine similarity, are the top five.
> basic_rag hands those five to the generator. It works for many questions, but look at the red
> part: these are questions it should have refused. It answered every one of them, about 40% of
> all questions. That's the problem the rest of the project solves.

*References:* Reimers & Gurevych, Sentence-BERT (EMNLP 2019); Wang et al., MiniLM (NeurIPS 2020);
Johnson et al., FAISS (2019).

### 4:18–5:28 · C: word-based retrieval and Graph RAG

**Show:** `theory_sparse_graph` (4:18–4:43), then `results/figures/video/graph_example.png`, then
`results/figures/video/retriever_heatmap.png`, then `results/figures/video/coronavirus_quality.png`.

> Hi, I'm Nur Hapinder, and I built Graph RAG. Instead of vectors, it matches exact words.
> BM25 scores a passage by its question words, weighted by how rare each word is, and stops
> rewarding a word once it's repeated. My graph links passages to the words in them. A question
> finds passages through its words, then takes one extra hop through rare words only, so it
> doesn't spread everywhere.
> Take "What is MVO?". The rare word "mvo" leads straight to the two right passages. MiniLM found
> neither; it doesn't know the abbreviation.
> I compared four retrievers. No one wins everywhere: MiniLM is best on the NIH pages, but on
> the COVID papers, the word-based methods are far better. And that carries into the answers:
> on coronavirus, graph_rag's answers contain the most of the true answer.

*References:* Robertson & Zaragoza, BM25 (2009); Edge et al., GraphRAG (2024).

### 5:28–6:30 · D: the generator

**Show:** `theory_bart` (5:28–5:53), then code `src/evrag/generator.py` lines 116–150 (`generate`,
`fit_contexts`), then `results/figures/video/closed_vs_open_book.png`.

> Hi, I'm Zhan Luming, and I trained the generator. Big models like ChatGPT weren't allowed, so
> we fine-tuned BART-base, about 140 million parameters. Its encoder reads the question and the
> passages; its decoder writes the answer word by word, looking back at them. We trained it on
> 66 thousand medical question-answer pairs to predict each next word of the right answer.
> All five methods share this one frozen model, so any difference comes from the evidence, not
> the generator.
> This chart shows why retrieval matters. With no documents, blue, it gets only 10 to 17% of the
> true answer. With retrieved passages, orange, that rises to 36 to 47%. A small model can't
> remember medicine, but it can read.

*Reference:* Lewis et al., BART (ACL 2020).

### 6:30–7:48 · E: when to abstain, and what we found

**Show:** `theory_abstain` (6:30–7:00), then code `src/evrag/methods/evidence_rag.py` lines 94–112
(`decide`), then `results/figures/reliability.png`, then `results/figures/abstention_by_scenario.png`.

> And I'm Yan Yuxuan again. I built the two methods that can say no, the evaluation and the demo.
> evidence_rag uses two signals: the match score, and coverage, which is how much of the
> question's important words the passage contains. If either is too low, it adds feedback words
> to the query and searches once more, and if that fails too, it abstains. The two thresholds
> are tuned on the dev set only, to catch the most unanswerable questions while refusing the
> fewest good ones.
> The result: the methods that can say no are about 25 points more reliable on every dataset,
> without losing correct answers.
> Two honest findings. evidence_rag ties with the simpler threshold_rag; it doesn't beat it.
> And when the right passage is missing, a similar passage on the same disease still scores
> high, so both methods sometimes answer anyway. That's the hardest case, and our next step.

*References:* Rocchio, relevance feedback (1971); Youden, Youden's J (1950); Kamath et al.,
Selective Question Answering under Domain Shift (ACL 2020).

### 7:48–8:13 · Wrap-up (E)

**Show:** `results/figures/video/pipeline.png` (7:48–8:03), then an end card with every reference
above and the GitHub repo link (last 10 s, no narration needed).

> To sum up: retrieval makes a small model useful, and letting it say "I don't know" makes it far
> more reliable. The same idea fits any field where a wrong answer is costly, like law or finance.
> Thanks for watching!

---

## Rendering and swapping in footage

[`configs/video_timeline_v2.yaml`](../configs/video_timeline_v2.yaml) holds this script as a timeline
(same times, speakers, narration and references); [`scripts/build_video.py`](../scripts/build_video.py)
renders it. Swapping in material works as in v1 (see "How to swap in footage" in
[video_script.md](video_script.md)): set `audio`, `camera` or a `footage` item in the block, then
render again.

```bash
python scripts/build_video.py --timeline configs/video_timeline_v2.yaml           # draft_v2.mp4
python scripts/build_video.py --timeline configs/video_timeline_v2.yaml --final   # final_v2.mp4
```

- **No generated voice in the submission.** Blocks without `audio` use a macOS `say` scratch voice,
  for timing only, and carry a red "DRAFT – scratch voice" mark. `--final` refuses to render while
  any block still uses the scratch voice, has no `camera`, or has an empty `footage` slot, and lists
  them.
- Output in `build/video/`: `draft_v2.mp4`, `draft_v2.srt`, `contact_sheet_v2.png` (two or three
  frames per block, camera box outlined in red).

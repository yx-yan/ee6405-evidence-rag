# Video script (9:55)

**Part 1 (0:00–4:15)** is a full demo walkthrough, narrated by E.
**Part 2 (4:15–9:55)** has each member explain their part: A → B → C → D → E.

## Before recording

```bash
python scripts/video_figures.py   # -> results/figures/video/*.png
streamlit run app/app.py          # demo at http://localhost:8501
```

- Run every demo example once before recording, so the models are loaded and the answers come
  back fast on camera.
- **Figures** come from two folders:
  - `results/figures/video/`: pipeline, datasets, basic_rag_errors, closed_vs_open_book,
    graph_example, retriever_heatmap, coronavirus_quality.
  - `results/figures/`: the notebook figures (reliability, abstention_by_scenario, risk_coverage,
    evidence_actions, recall_at_k).
  - All twelve are 1920x1080 (16:9) in one style (`src/evrag/figures.py`). The top-right corner
    (right 26% of the width, top 40% of the height) is left empty for the presenter's camera;
    `python scripts/video_figures.py --guide` outlines it to check the placement.
- **Code shots**: open the file in the editor at the given lines, with the font large enough to read
  on video.
- **Pace**: about 150 words a minute. Each block below is timed to that, so read at a normal speed.
- **References**: each part lists its references. Show them as a small caption in the bottom
  corner of the screen while that part plays; they are not read aloud.
- Each member starts their part by saying their name, as written in the script.

---

## Part 1 — Demo (E)

### 0:00–0:30 · The problem

**Show:** the demo's start page (title "Evidence-aware RAG").
**Action:** while talking, move the mouse over the *Dataset* buttons, the *Method* buttons and the
collapsed sidebar (explanation settings), so viewers see what can be chosen.

> Hi, we're Group X, and I'm Yan Yuxuan. Imagine asking a medical chatbot a question it has no information about.
> Most chatbots built on retrieval will still answer, and they'll sound confident. In medicine
> that's dangerous. So our project asks a simple question: when should the system answer, when
> should it go and look again, and when should it just say "I don't know"? Let me show you.

### 0:30–1:10 · Five methods, one pipeline

**Show:** `results/figures/video/pipeline.png`

> We compare five methods, and they all share the same parts: the same retriever, and the same
> small generator, a BART model we fine-tuned ourselves. No ChatGPT anywhere.
> The first three always answer. nlp_only uses no documents at all. basic_rag looks up the
> top five passages. graph_rag finds passages through a graph of shared words.
> The last two are allowed to say no. threshold_rag abstains when even the best passage scores
> too low. evidence_rag checks two things, how well the passage matches and how much of the
> question it covers. If that's not enough, it searches once more with a better query, and only
> then gives up.

*References:* Lewis et al., Retrieval-Augmented Generation (NeurIPS 2020); Lewis et al., BART
(ACL 2020).

### 1:10–1:45 · Example 1: it answers

**Show:** demo → click **Answerable**, dataset *cancer*, method *evidence_rag* → **Generate**.

> Here's a normal question: "What are the symptoms of gallbladder cancer?"
> These two numbers are the evidence check. The match score is 0.79, and the bar is 0.71, so
> it passes. Coverage is 1.0, which means every important word in the question shows up in the
> passage. Both pass, so it answers, and the answer lists the symptoms, starting with jaundice.
> Down here you can see exactly which passages it used.

### 1:45–2:25 · Example 2: it looks again

**Show:** click **Retrieve again** → **Generate**. Point at the grey code box with the expanded query.

> Now: "Who is at risk for testicular cancer?" The first search scores 0.696, just under the
> bar. Instead of giving up, it takes the top passages, picks the most telling words from them,
> like "undescended" and "cryptorchidism", and adds them to the question. Second try: 0.77,
> so it passes and answers. This is the "retrieve again" step from our title.

### 2:25–3:10 · Example 3: the answer isn't in the database

**Show:** click **Missing evidence**. First switch the method to **basic_rag** → **Generate**,
then switch back to **evidence_rag** → **Generate**.

> For this question we deliberately deleted the right passages from the database:
> "What is prostate cancer?"
> Watch basic_rag first. It answers anyway, with something about Gleason scores and cancer
> spreading to the bladder. It sounds medical, but it doesn't answer the question.
> Now evidence_rag. First search: 0.65, too low. It tries again: 0.68, still under 0.71. So it
> says "not enough evidence". That's the behaviour we want.

### 3:10–3:30 · Example 4: wrong topic

**Show:** click **Out of domain** → **Generate**.

> And if you ask our cancer knowledge base about COVID symptoms? The match is 0.39 and coverage is
> only 0.2. It retries, still fails, and abstains, instead of making up a cancer answer.

### 3:30–4:15 · Graph route and "why did it say that?"

**Show:**
1. Dataset *coronavirus*, method **graph_rag**, type `What is MVO?` → **Generate**. Open the
   expander "How the graph connected the question to evidence".
2. Click **Distractor**, scroll to the coloured heatmap.

> Two more things. Here's graph_rag on the COVID papers. "MVO" is a rare abbreviation; the graph
> links it straight to the two passages that define it. Our teammate will explain this later.
> Last, every answer can be explained. Here we pasted an asthma passage and an eczema passage
> and asked about eczema. Red means "this word pushed the answer", so you can check which
> passage the answer really came from. We compute this with three methods: SHAP, LIME
> and occlusion. Now let's hand over to the team to show how each part was built.

*References:* Lundberg & Lee, SHAP (NeurIPS 2017); Ribeiro et al., LIME (KDD 2016).

---

## Part 2 — Who did what

### 4:15–5:05 · A: the data

**Show:** `results/figures/video/datasets.png`, then the code at
`src/evrag/data/disease_sets.py` lines 1–25 (the docstring table) and `chunk_spans` at lines 56–61.

> Hi, I'm Nguyen Khanh Nam, and I built the data. We wanted three different diseases from three different sources, so the
> results don't depend on one dataset. Coronavirus comes from COVID-QA, which uses research
> papers. Cancer and neurological disorders come from MedQuAD, which uses NIH patient pages.
> Long texts are cut into 150-word passages that overlap by 30 words, so an answer is never
> sliced in half at a border.
> Every set has three kinds of test question. Answerable ones are the blue part. For the orange
> part, missing evidence, we removed the right passages. The green part, out of domain, asks
> about a different disease. So almost half the test questions should not be answered at all.

*References:* Möller et al., COVID-QA (2020); Ben Abacha & Demner-Fushman, MedQuAD (BMC
Bioinformatics 2019).

### 5:05–5:50 · B: retriever and basic RAG

**Show:** code `src/evrag/retriever.py` lines 114–140 (`search`) and
`src/evrag/methods/basic_rag.py`, then `results/figures/video/basic_rag_errors.png`.

> Hi, I'm Xu Yiwen. I built the shared retriever and basic RAG. The retriever turns every passage into a vector
> with a small model called MiniLM. A question becomes a vector too, and we take the five
> closest passages. basic_rag just feeds those to the generator.
> It's simple, and it works on many questions. But look at this chart. The red part is
> questions it should have refused, and it answered every one, about 40% of all questions.
> basic_rag never says no. That's the problem the rest of the project is about.

*References:* Reimers & Gurevych, Sentence-BERT (EMNLP 2019); Wang et al., MiniLM (NeurIPS 2020);
Johnson et al., FAISS (2019).

### 5:50–7:20 · C: Graph RAG and the retriever comparison

**Show:**
1. Code `src/evrag/methods/graph_rag.py` lines 88–152 (`search`: direct matches, then one hop).
2. `results/figures/video/graph_example.png`.
3. `results/figures/video/retriever_heatmap.png`.
4. `results/figures/video/coronavirus_quality.png`.

> Hi, I'm Nur Hapinder, and I built Graph RAG. Instead of vectors, it builds a graph. Every passage is connected to the
> words in it. A question lights up its words, the words lead to passages, and then we take one
> extra hop: passages that share rare words with those passages get pulled in too. Common words
> are ignored for the hop, so it doesn't spread everywhere.
>
> Here's a real test question: "What is MVO?" The rare word "mvo" leads straight to the two
> correct passages, the green ones. MiniLM, on the same question, found none of them. It doesn't
> know what the abbreviation means.
>
> I also compared four retrievers on their own: two classic ones, TF-IDF and our own BM25, the MiniLM model, and my graph. This shows how often the right passage is in
> the top five. The interesting part: no one wins everywhere. On the NIH pages MiniLM is best,
> at 0.91 and 0.98. But on the COVID research papers it drops to 0.43, while the word-based
> methods, including the graph, stay around 0.78.
>
> That carries through to the answers. On coronavirus, graph_rag gets the most of the true
> answer into its reply, 0.55, against 0.38 for basic RAG. So for technical papers full of
> abbreviations, matching exact words still beats embeddings.

*References:* Robertson & Zaragoza, BM25 (2009); Edge et al., GraphRAG (2024), the idea our
lighter term graph follows.

### 7:20–8:20 · D: the generator

**Show:** code `src/evrag/generator.py` lines 116–150 (`generate`, `fit_contexts`), then
`results/figures/video/closed_vs_open_book.png`.

> Hi, I'm Zhan Luming, and I trained the generator. The course didn't allow big models like ChatGPT, so we fine-tuned
> BART-base, about 140 million parameters, on medical question-answer data. It reads the
> question plus the passages and writes the answer. fit_contexts makes sure the passages fit
> into its 1,024-token window. All five methods use this same frozen model, so any difference
> comes from how they pick evidence, not from the generator.
>
> This chart shows why retrieval matters. The blue bars are the same model with no documents:
> it covers only 10 to 17% of the true answer. Give it the retrieved passages and that jumps to
> 36 to 47%. A small model can't remember medicine, but it's good at reading.
>
> One honest note: part of our training data came from the same NIH websites, so we checked the
> overlap. Removing the questions it might have seen changes the results by about 0.02.

*Reference:* Lewis et al., BART (ACL 2020).

### 8:20–9:35 · E: deciding when to abstain, and what we found

**Show:**
1. Code `src/evrag/methods/evidence_rag.py` lines 44–52 (`coverage`) and 94–112 (`decide`).
2. Code `src/evrag/evaluation/tune.py` lines 44–50 (`youden`).
3. `results/figures/reliability.png`.
4. `results/figures/abstention_by_scenario.png`.

> And I'm Yan Yuxuan again. I built the two methods that can say no, plus the evaluation and the demo. evidence_rag
> checks the match score and the question coverage, retries once with an expanded query, and
> abstains if the checks still fail. The thresholds are tuned only on the dev set, never on
> test. We
> pick the one that catches the most unanswerable questions while refusing the fewest good ones.
>
> Our main measure is reliability: how often the system does the right thing, meaning a correct
> answer or a correct refusal. Here, the methods that can say no gain about 25 points on every
> dataset, without losing correct answers, and a paired bootstrap confirms the gain is real.
>
> Now the honest part. evidence_rag ties with the simpler threshold_rag; it doesn't beat it.
> And this chart shows where both still fail. Off-topic questions get caught almost every time,
> but when the right passage is missing, a similar passage on the same disease is still there,
> scores almost as high, and the system answers anyway. That's the hardest case, and the clear
> next step.

*References:* Rocchio, relevance feedback (1971); Youden, Youden's J (1950); Kamath et al.,
Selective Question Answering under Domain Shift (ACL 2020); Geifman & El-Yaniv, risk-coverage
(NeurIPS 2017).

### 9:35–9:55 · Wrap-up (E)

**Show:** `results/figures/video/pipeline.png` again, or the README results table.

> So, to sum up: retrieval is what makes a small model useful, and letting the system say "I
> don't know" makes it far more reliable. Telling "no answer here" apart from "a similar answer
> here" is still open. Everything is in our GitHub repo, with a notebook to reproduce every
> number. Thanks for watching!

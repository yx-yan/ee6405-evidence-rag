# Shared answer generator (owner: D)

The three routes we compare (no retrieval, standard RAG, Graph RAG) all answer with the same
model; only the evidence passed to it differs. The model is **BART-base fine-tuned by us on
general medical QA**. Code: `src/evrag/generator.py` (inference), `src/evrag/data/generator_data.py`
(training data), `scripts/train_generator.py` (training), `scripts/train_generator.sbatch` (SLURM).

```python
from evrag import generator
generator.generate("What are the symptoms of eczema?")                 # route 1: no evidence
generator.generate("What are the symptoms of eczema?", contexts=docs)  # routes 2 and 3: retrieved text or RetrievedDoc objects
```

**Getting the model.** The trained weights are not in git. They are attached to the GitHub Release
[generator-v1](https://github.com/yx-yan/ee6405-evidence-rag/releases/tag/generator-v1)
(`bart-base-best.zip`, 0.5 GB). The first `generate()` call downloads and unzips them into
`models/generator/bart-base/best` (`generator.model_dir`) if that folder is missing. This needs the
GitHub CLI logged in once (`gh auth login`). Without `gh`, download the zip from the link and unzip
it into that folder yourself. To use another checkpoint, point `generator.model_dir` (or
`EVRAG_GENERATOR_DIR`) at its directory.

## Why BART-base

BART is an encoder–decoder pretrained as a whole (denoising), so it can write fluent answers after
fine-tuning on tens of thousands of QA pairs. A BERT2BERT model would start with randomly
initialised cross-attention and need more data and training to generate well. BART-base has 140M
parameters (an LLM has billions); BART-large (400M) is the fallback if base is too weak.

## Training data

General medical QA, kept apart from the RAG knowledge base and its evaluation questions:

| Source | Content | Examples (train) | Licence / note |
|---|---|---|---|
| `wikidoc_patient` — medalpaca/medical_meadow_wikidoc_patient_information | WikiDoc patient-information Q&A ("What are the symptoms of X?") | 5,152 closed | CC; questions are the page's own subheadings |
| `wikidoc_textbook` — medalpaca/medical_meadow_wikidoc | WikiDoc living-textbook paragraphs | 8,042 closed | CC; questions were rephrased from headings by GPT-3.5 (the card reports ~30% weak rephrasings); drop with `--sources` if unwanted |
| `mashqa` — MASH-QA (Zhu et al., Findings of EMNLP 2020) | WebMD consumer questions, answers are sentences of the WebMD article | 27,540 closed + 24,899 context | research dataset from the authors' Google Drive (github.com/mingzhu0527/MASHQA) |
| `mediqa` — medalpaca/medical_meadow_mediqa (MEDIQA-AnS) | consumer questions, source sections, summary answers | 836 context | NIH-derived sources; check overlap once the RAG sources are known |

Totals: train 66,469 / val 7,643 / test 7,451 (MASH-QA keeps its official splits; the others are
split by a hash of the question). Each example is `closed` (question → answer) or `context`
(question + 4 evidence chunks → answer). Context chunks mimic retrieval: the article is cut into
≤150-word chunks and the chunks holding the answer are mixed with other chunks of the same article
and of other articles, in random order. Questions in val/test never appear in train.

**Keeping the RAG evaluation clean.** Once the RAG questions exist, rebuild with
`python -m evrag.data.generator_data --exclude <questions.csv>` (a CSV with a `question` column, or
one question per line); those questions are then dropped everywhere before training.

Input format (`format_input`): `question: <q>` or `question: <q> context: [1] <chunk> [2] <chunk> …`.
The question comes first, so truncation at 1,024 tokens only cuts evidence.

## Build and train

```bash
pip install -r requirements-generator.txt && pip install -e .   # after a CUDA build of torch
python -m evrag.data.generator_data          # ~15 s after downloads; writes data/generator/ (not committed)
python scripts/train_generator.py --out models/generator/bart-base --bf16      # interactive GPU
sbatch scripts/train_generator.sbatch        # SLURM (written for cluster02; other clusters add partition/account)
```

- Compute nodes without internet: run the data step once on a login node, and pre-download the
  base model there (`python -c "from transformers import AutoModelForSeq2SeqLM as M, AutoTokenizer as T; M.from_pretrained('facebook/bart-base'); T.from_pretrained('facebook/bart-base')"`
  with the same `HF_HOME`).
- MASH-QA without gdown: download the archive from the Google Drive link in the MASH-QA README and
  save it as `data/raw/generator/mashqa/mashqa.zip`.
- Defaults: 3 epochs, batch 16, lr 5e-5, linear warm-up 3%, bf16 on supported GPUs. About 12.5k
  steps; expect roughly an hour on one modern GPU.
- Progress: one JSON line every 50 steps (loss, lr, speed, ETA) in the job log and
  `<out>/log.jsonl`; validation loss and ROUGE-L (closed / context) every 2,000 steps and per epoch.
- Re-running with `--resume` (the sbatch script does this automatically) continues from `<out>/last`.
- Outputs: `<out>/best` (lowest validation loss; point `generator.model_dir` here), `<out>/last`,
  `<out>/test_eval.json` (ROUGE-L on 1,000 test examples with the shared beam-search settings) and
  `<out>/test_samples.jsonl` (predictions to read).

## Results (bart-base, first version)

Trained on one RTX PRO 6000 (bf16, 3 epochs, 14 minutes). Validation loss 1.697 → 1.561.
Test ROUGE-L (1,000 test examples, shared beam settings): **closed 0.136 / context 0.528**.

| Source | Closed | Context |
|---|---|---|
| mashqa | 0.131 (n=423) | 0.531 (n=491) |
| mediqa | – | 0.373 (n=9) |
| wikidoc_patient | 0.173 (n=38) | – |
| wikidoc_textbook | 0.151 (n=39) | – |

What to expect when using it:

- **Closed (route 1)** answers are fluent and on topic but short (median 24 words vs 54 in the
  references) and sometimes medically wrong or repetitive; the model only knows what fits in 140M
  parameters.
- **With evidence (routes 2 and 3)** it mostly copies the relevant sentences from the chunks and
  ignores distractors (80 MASH-QA test questions: gold + distractors 0.56, gold only 0.57).
- It **never says "I don't know"**: training always included the answer in the evidence, so with
  evidence that lacks the answer it still copies from it (ROUGE-L 0.11, below closed 0.13).
  Retrieval quality therefore matters a lot for routes 2 and 3.

ROUGE-L here is only a sanity check that the model learned to answer and to use evidence; the
route comparison is evaluated on the RAG questions.

CPU smoke test (no GPU needed): `python scripts/train_generator.py --out models/smoke --limit 48 --max-steps 4 --batch-size 4 --eval-every 2`.

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

Set `generator.model_dir` in `configs/default.yaml` (or `EVRAG_GENERATOR_DIR`) to the trained
checkpoint directory (`.../best`). Checkpoints are not committed.

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
sbatch scripts/train_generator.sbatch        # SLURM: fill in partition/account first
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

ROUGE-L here is only a sanity check that the model learned to answer and to use evidence; the
route comparison is evaluated on the RAG questions.

CPU smoke test (no GPU needed): `python scripts/train_generator.py --out models/smoke --limit 48 --max-steps 4 --batch-size 4 --eval-every 2`.

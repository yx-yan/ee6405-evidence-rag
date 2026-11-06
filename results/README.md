# Results

| Folder | Content | Written by |
|--------|---------|------------|
| `runs/` | Method runs in the `run_experiment` format. The `*_medquad_*` files are the MedQuAD trial with MiniLM and `gpt-4.1-mini`. | `evrag.evaluation.run_experiment`, `scripts/d_run_medquad_trial.py` |
| `llm_medquad/` | The same MedQuAD trial with `bge-base-en-v1.5` and Qwen. | `scripts/d_run_medquad_trial.py` |
| `d_medquad_trial/` | Grading, label audit and embedding comparison ([report](../docs/reports/d_medquad_trial.md)). | D |
| `llm_screen/` | LLM screen on MTS-Dialog ([report](../docs/llm_selection.md)). | `scripts/llm_screen.py` |
| `dataset_knowledge/` | Prior-knowledge probes on the three datasets ([report](../docs/llm_selection.md)). | `scripts/dataset_knowledge.py` |
| `b_validation/` | Baseline validation records ([report](../docs/reports/b_live_validation.md)). | `scripts/b_*.py` |
| `figures/` | Figures for the video. | `evrag.evaluation.plots` |

`runs/` and `llm_medquad/` share file names but hold different results. `scripts/d_run_medquad_trial.py` always writes to `runs/`, so rerunning it overwrites the `gpt-4.1-mini` files.

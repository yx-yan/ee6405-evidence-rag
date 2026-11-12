# Results

| Folder | Content | Written by |
|--------|---------|------------|
| `runs/` | Method runs in the `run_experiment` format. | `evrag.evaluation.run_experiment` |
| `figures/` | Figures for the video. | plotting notebooks |
| `metrics.csv` | Metrics per method, per scenario when the dataset has one, and for all questions. | `python -m evrag.evaluation.metrics --split test` |
| `adjudication_<split>.csv` | Stratified sample of answers to label by hand (`label`: correct / incorrect); labels override the automatic F1 rule. | `... metrics --adjudication`, then people |

Earlier LLM screens and MedQuAD trials were removed on 2026-11-08 and are in git history.

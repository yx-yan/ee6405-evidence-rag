## What & why

<!-- One or two sentences. Link the issue: Closes #… -->

## Checklist

- [ ] `pytest` passes locally (CI will check too)
- [ ] Uses the shared `retriever` / `llm` and `configs/default.yaml`, with no hard-coded model names or `top_k`
- [ ] Thresholds and prompts tuned on **dev** only; test split untouched
- [ ] `schema.py` / `base.py` unchanged, or the change is explained here for E
- [ ] Method PRs: 20-question smoke run attached (`results/runs/<method>_dev20.jsonl`)
- [ ] AI tools used? If yes, what for and the shareable chat link:

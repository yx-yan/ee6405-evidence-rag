# Contributing

## Branches & pull requests

- `main` is always runnable and protected: changes go in through a pull request with one approval and passing CI.
- Branch per task, prefixed with your role: `a-corpus`, `b-retriever`, `c-threshold`, `d-judge`, `e-metrics`.
- Keep PRs small (one feature each), and pull `main` often (`git pull origin main`).
- PRs are squash-merged.

## Interface contract

- Every method subclasses `evrag.methods.base.Method`, implements `_answer()`, and is registered with `@register`.
- `run()` returns `evrag.schema.RAGOutput`. **Do not change `schema.py` or `base.py` without E's approval** — evaluation and the demo depend on them.
- `tests/test_contract.py` checks every registered method automatically.

## Experimental rules

- Read model names, `top_k` and the seed from `configs/default.yaml`. Never hard-code them.
- Tune thresholds and prompts on the **dev** split only. The **test** split is for final numbers.
- The LLM is called only through `evrag.llm`; retrieval only through `evrag.retriever`.
- A method PR includes a 20-question smoke run: `results/runs/<method>_dev20.jsonl`.

## Commits

- Small, focused commits with clear messages, e.g. `c: tune threshold on dev split`.
- Commit your own work under your own GitHub account: the history is our evidence of contribution for the group reflection.

## Code & notebooks

- Reusable logic lives in `src/evrag/`. Notebooks only run experiments and plot.
- Notebook names: `<role>_<topic>.ipynb`, e.g. `C_threshold_tuning.ipynb`. Clear large outputs before committing.
- Docstrings on public functions; no absolute paths (use `evrag.config`).

## Data & secrets

- `data/processed/` (the frozen eval set) is committed. `data/raw/`, model weights and `.env` are not.
- Never commit API keys.

## AI tools

Our submission includes an AI disclosure form. If you use an AI tool (ChatGPT, Claude, Copilot, ...), keep the shareable chat link and note what you used it for in the PR. Understand and be able to explain every line you commit.

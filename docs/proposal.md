# Proposal (summary)

Full interactive version: [proposal.html](proposal.html) (download and open locally).

## Research question

Can an evidence-aware RAG workflow **reduce unsupported answers without rejecting too many answerable questions**?

Traditional RAG answers even when the retrieved documents are irrelevant, incomplete or misleading. A trustworthy system should sometimes say *"I don't have enough evidence."* Our contribution is a reproducible comparison of answer-versus-abstain trade-offs across evidence quality. We do not claim to invent RAG abstention.

## Three methods, one shared backend

The knowledge base, embeddings, top-k, LLM, generation settings and question set are the same for all three methods. **Only the response policy changes.**

| # | Method | Policy | Hypothesis |
|---|--------|--------|------------|
| 1 | Standard RAG | retrieve top-k → prompt LLM → answer | high coverage, some unsupported answers |
| 2 | Threshold RAG | abstain if relevance score < threshold tuned on dev | fewer unsupported answers, may over-refuse |
| 3 | Evidence-aware RAG | LLM judges SUFFICIENT / INSUFFICIENT → answer / abstain; optional one re-retrieval | better-targeted refusals, more latency/cost |

Keep it lightweight: frozen LLM and embeddings, no fine-tuning, no multi-agent orchestration. Re-retrieval is optional.

## Data: SQuAD 2.0, three evidence scenarios

150–300 evaluation questions total, with a fixed retrieval corpus.

| Scenario | Construction | Desired behaviour |
|----------|--------------|-------------------|
| A · Sufficient | answer supported by an indexed passage | accurate answer + evidence |
| B · Missing | supporting passages withheld / unanswerable from the corpus | explicit abstention |
| C · Distracting | related passages lacking the fact (optionally contradictory) | no invented definitive answer |

A "no answer" label is only valid relative to the **whole** searchable corpus. Deduplicate, avoid leakage, audit every scenario manually.

## Metrics

| Metric | Definition | Direction |
|--------|------------|-----------|
| Answer correctness | correct supported answers / all questions | ↑ |
| Selective accuracy | correct answers / answered questions | ↑ (with coverage) |
| Coverage | answered / all | context-dependent |
| Abstention recall | correctly refused insufficient-evidence Qs / all insufficient-evidence Qs | ↑ |
| Over-refusal rate | refused sufficient-evidence Qs / all sufficient-evidence Qs | ↓ |
| Unsupported-answer rate | answers not substantiated by retrieved evidence / generated answers | ↓ |
| Latency | mean and p95 per question | ↓ |

Never call a method "better" just because it refuses almost everything.

## Planned figures

1. Stacked correct / incorrect / abstained per method, by scenario
2. Selective accuracy vs coverage as the threshold changes (test split)
3. Latency vs reliability

## Interface contract

```python
method.run(question) -> RAGOutput(question_id, method, answer | None, abstained,
                                  retrieved_docs, latency_ms, meta)
```

See `src/evrag/schema.py`.

## Key references

- Joren et al., *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems*, ICLR 2025
- Peng et al., *Unanswerability Evaluation for Retrieval Augmented Generation (UAEval4RAG)*, ACL 2025
- Rajpurkar, Jia & Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD*, ACL 2018

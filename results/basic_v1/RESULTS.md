# Basic RAG v1 results

Fixed-configuration comparison on A's WikiDoc data (data/processed).

| Split | Method | Questions | Token F1 | ROUGE-L | Automatic correctness | Retrieval hit | Mean latency ms |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | basic_rag | 175 | 0.3897 | 0.3587 | 0.3143 | 0.8343 | 2427.8504 |
| dev | nlp_only | 175 | 0.1629 | 0.1211 | 0.0286 | N/A | 1481.1096 |
| test | basic_rag | 175 | 0.3773 | 0.3356 | 0.2743 | 0.8114 | 2342.7084 |
| test | nlp_only | 175 | 0.1751 | 0.1310 | 0.0286 | N/A | 1346.8411 |

Automatic correctness uses token F1 >= 0.5. These values have not been manually adjudicated.

The checkpoint may overlap the corpus in training, and three normalized questions occur in both splits.

All questions have gold answers, so abstention recall is N/A. Evidence-support heuristics cannot establish factual correctness.

Latency includes first-call model/index loading and reflects CPU runs on this machine.

Read docs/reports/basic_rag_v1.md for setup and source provenance.

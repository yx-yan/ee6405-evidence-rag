# Basic RAG v1 results / 结果

Fixed-configuration comparison on A's WikiDoc data (data/processed).
A 的 WikiDoc 数据（data/processed）上的固定配置对比。两种方法共生成 700 条输出，配对覆盖 350 道题。

| Split | Method | Questions | Token F1 | ROUGE-L | Automatic correctness | Retrieval hit | Mean latency ms |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | basic_rag | 175 | 0.3897 | 0.3587 | 0.3143 | 0.8343 | 2427.8504 |
| dev | nlp_only | 175 | 0.1629 | 0.1211 | 0.0286 | N/A | 1481.1096 |
| test | basic_rag | 175 | 0.3773 | 0.3356 | 0.2743 | 0.8114 | 2342.7084 |
| test | nlp_only | 175 | 0.1751 | 0.1310 | 0.0286 | N/A | 1346.8411 |

Automatic correctness uses token F1 >= 0.5. These values have not been manually adjudicated.
自动正确率使用 Token F1 >= 0.5，尚未人工判分。

The checkpoint may overlap the corpus in training, and three normalized questions occur in both splits.
现有权重可能在训练时见过语料，dev/test 有三个归一化后的重复问题。

All questions have gold answers, so abstention recall is N/A. Evidence-support heuristics cannot establish factual correctness.
所有题目都有参考答案，拒答召回率为 N/A。词汇证据支持启发式不能证明事实正确。

Latency includes first-call model/index loading and reflects CPU runs on this machine.
延迟包含各次运行首题的模型/索引加载，反映当前电脑 CPU 运行情况。

Read docs/reports/basic_rag_v1.md for setup and source provenance.
环境设置及来源追溯见 docs/reports/basic_rag_v1.md。

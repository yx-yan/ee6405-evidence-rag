# Basic RAG v1

## 中文说明

这个实验将 Nam（A）的 WikiDoc 数据（data/processed，见 data/README.md）、Jojo 的共享 BART 生成接口、Basic RAG 和 E 的评估代码接在一起。数据包括 512 个段落和 350 道问题，dev/test 各 175 道。所有段落不超过 150 个空格分隔的词。

NLP-only 只将问题送给 BART。Basic RAG 用 all-MiniLM-L6-v2 编码问题和语料，归一化向量后通过 FAISS IndexFlatIP 搜索 top-5。按排名选取最多 4 段（configs/default.yaml 的 retrieval.context_k），并用 BART tokenizer 检查完整输入是否在 1,024 tokens 内。两种方法使用同一份微调 BART 权重和相同生成参数。检索为空时 Basic 返回 answer=null、abstained=true，跳过生成。有检索结果时直接生成，没有另加阈值或答案充分性判断。

当前词汇正确率由 E 的 Token F1 >= 0.5 规则计算，没有人工判分。support 是词汇重合启发式，不能证明答案事实正确。这里让它使用实际传给 BART 的证据，NLP-only 的证据支持率记为 N/A。没有 scenario 标签时仅输出 all 整体指标。

数据全为可回答题，拒答召回率记为 N/A。三个归一化后的问题同时出现在 dev 和 test。当前 BART 与语料可能有训练重叠，因此这是草稿数据上的固定配置比较。没有基于这些结果调参。本地 BART ZIP 与 Release 公布的压缩包 SHA-256 不一致，但其中的 model.safetensors 与 Release 压缩包内的完全相同（2026-11-12 核对），记录见 results/basic_v1/run_manifest.json。

### 关键文件

- src/evrag/methods/basic_rag.py 将检索和生成接起来。
- src/evrag/methods/nlp_only.py 用同一个 BART 直接回答问题。
- src/evrag/retriever.py 读取语料并缓存语义向量。
- src/evrag/generator.py 是队友已有的 BART 推理代码。
- src/evrag/evaluation/run_experiment.py 批量运行并保存逐题 JSONL。
- src/evrag/evaluation/metrics.py 是 E 的共享评估代码（PR #29），已包含上述整体分组与证据支持计算调整。
- scripts/summarize_basic.py 核对输出、证据文本、输入长度，并导出逐题配对结果及来源表。
- scripts/run_basic_v1.sh 运行完整 dev/test 的两种方法和评估。

### 运行

在项目根目录创建 Python 3.11 虚拟环境，安装依赖。torch 需要按运行设备安装合适版本。

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install torch
    pip install -r requirements.txt -r requirements-generator.txt
    scripts/run_basic_v1.sh

脚本使用 models/generator/bart-base/best 内的 BART 权重（首次调用会从 GitHub Release 下载）；需要使用其他 BART 时，可通过 EVRAG_GENERATOR_DIR 指定其目录。MiniLM 由 sentence-transformers 自动下载。所有设置读自 configs/default.yaml。脚本会重新生成同名 run 和指标文件，重跑前可备份 results。

### 查看输出

- results/basic_v1/runs/nlp_only_dev.jsonl 和 nlp_only_test.jsonl 是问题单独输入 BART 的结果。
- results/basic_v1/runs/basic_rag_dev.jsonl 和 basic_rag_test.jsonl 是检索后生成的结果。
- results/basic_v1/metrics_dev.csv 和 metrics_test.csv 是整体自动指标。空白值表示该项无法计算。
- results/basic_v1/paired_comparison.csv 对齐每题的两种答案、F1、ROUGE-L 和证据命中数。
- results/basic_v1/context_sources.csv 给出实际送入 BART 的证据及原始来源字段。来源字段沿用 Nam 数据，字段值不一定是可定位的原始网页。
- results/basic_v1/run_manifest.json 记录数据摘要、文件校验和及限制。
- results/basic_v1/full_run.log 记录批量进度。

逐题 JSONL 中 retrieved_docs 包含 top-5 候选，meta.context_doc_ids 标识实际输入 BART 的段落。两者可能不同。现有 recall_at_k 指标是至少命中一个 gold ID 的比例。多证据题可查看 paired_comparison.csv 的 gold_fraction_retrieved、all_gold_retrieved 和 context_gold_count。

## English explanation

This experiment connects Nam's (A) WikiDoc data (data/processed, see data/README.md), Jojo's shared BART generator, Basic RAG, and E's evaluation code. The corpus has 512 passages, each containing at most 150 whitespace-delimited words. The dataset contains 350 questions with 175 each in dev and test.

NLP-only passes the question directly to the fine-tuned BART checkpoint. Basic RAG uses all-MiniLM-L6-v2 embeddings and normalized FAISS IndexFlatIP search to retrieve five candidates. It selects up to four passages (retrieval.context_k in configs/default.yaml) in rank order while checking the complete BART input against its 1,024-token budget. Both routes use the same generator and decoding parameters. Empty retrieval produces answer=null and abstained=true without invoking BART. Nonempty retrieval invokes BART without an additional sufficiency gate.

Automatic correctness uses E's token-F1 threshold of 0.5. No human labels were applied. Evidence support uses a lexical heuristic, scored against the contexts actually supplied to BART. Evidence support for NLP-only is N/A. Only the aggregate all group is produced when scenario labels are absent.

All questions have gold answers, so abstention recall is N/A. Three normalized questions occur in both splits. The existing checkpoint may overlap the corpus in its training data. The local checkpoint archive hash differs from the published Release archive hash, but its model.safetensors is identical to the one in the Release archive (checked 2026-11-12). These limitations are recorded in results/basic_v1/run_manifest.json. Parameters are fixed for this draft-data comparison.

Install the dependencies with the commands above and run scripts/run_basic_v1.sh. It uses the BART checkpoint in models/generator/bart-base/best (downloaded from the GitHub Release on first use); EVRAG_GENERATOR_DIR selects another checkpoint directory. sentence-transformers downloads MiniLM. Every setting is read from configs/default.yaml.

The results/basic_v1/runs directory contains the four raw run files. metrics_dev.csv and metrics_test.csv contain automatic metrics. paired_comparison.csv aligns both answers for every question and reports individual gold-passage recall and complete-gold retrieval. context_sources.csv links actual generator contexts to the corpus metadata. Source fields are inherited from the supplied data and may not identify a specific original webpage. Empty CSV metrics mean the measure is undefined for these inputs.

The metric named recall_at_k measures whether any gold passage was retrieved. For multi-evidence questions, also inspect gold_fraction_retrieved, all_gold_retrieved, and context_gold_count in paired_comparison.csv. retrieved_docs contains all five candidates, whereas meta.context_doc_ids lists the passages selected for generation.

The changes add Basic RAG and NLP-only methods, progress logging with flushed run output, aggregate evaluation without scenario labels, and paired-output auditing.

实验配置为共享的 configs/default.yaml，结果目录由 EVRAG_RESULTS_DIR 选择（默认 results/basic_v1）。
The experiment uses the shared configs/default.yaml; EVRAG_RESULTS_DIR selects the results folder (results/basic_v1 by default).

数据来自 Nam / Illgetgood（A）。评估使用 Yuxuan / yx-yan 的共享 metrics.py（PR #29），其中的无 scenario 分组和证据支持调整由 Yiwen Xu 提出。Jojo082099 提供共享 BART 生成器及训练权重。
Data is credited to Nam / Illgetgood (A). Evaluation uses the shared metrics.py by Yuxuan / yx-yan (PR #29), including the scenario-free grouping and evidence-support fixes proposed by Yiwen Xu. The shared BART generator and checkpoint are credited to Jojo082099.

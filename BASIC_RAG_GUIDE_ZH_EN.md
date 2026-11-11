# Basic RAG v1

## 中文说明

这个本地项目将 Nam 的 Illgetgood-patch-2 数据、Jojo 的共享 BART 生成接口、Basic RAG 和 E 的评估代码接在一起。数据包括 512 个段落和 350 道问题，dev/test 各 175 道。所有段落不超过 150 个空格分隔的词。

NLP-only 只将问题送给 BART。Basic RAG 用 all-MiniLM-L6-v2 编码问题和语料，归一化向量后通过 FAISS IndexFlatIP 搜索 top-5。按排名选取最多 4 段，并用 BART tokenizer 检查完整输入是否在 1,024 tokens 内。两种方法使用同一份微调 BART 权重和相同生成参数。检索为空时 Basic 返回 answer=null、abstained=true，跳过生成。有检索结果时直接生成，没有另加阈值或答案充分性判断。

当前词汇正确率由 E 的 Token F1 >= 0.5 规则计算，没有人工判分。support 是词汇重合启发式，不能证明答案事实正确。这里让它使用实际传给 BART 的证据，NLP-only 的证据支持率记为 N/A。没有 scenario 标签时仅输出 all 整体指标。

数据全为可回答题，拒答召回率记为 N/A。三个归一化后的问题同时出现在 dev 和 test。当前 BART 与语料可能有训练重叠，因此这是草稿数据上的固定配置比较。没有基于这些结果调参。本地 BART ZIP 与 Release 公布的压缩包 SHA-256 不一致，无法独立确认和 Release 原件完全相同。使用的是已成功加载的本地 safetensors 权重，记录见 results/basic_v1/run_manifest.json。

### 关键文件

- src/evrag/methods/basic_rag.py 将检索和生成接起来。
- src/evrag/methods/nlp_only.py 用同一个 BART 直接回答问题。
- src/evrag/retriever.py 读取语料并缓存语义向量。
- src/evrag/generator.py 是队友已有的 BART 推理代码。
- src/evrag/evaluation/run_experiment.py 批量运行并保存逐题 JSONL。
- src/evrag/evaluation/basic_metrics.py 来自 E 的 e-metrics 版本，并做上述整体分组与证据支持计算调整。
- scripts/summarize_basic.py 核对输出、证据文本、输入长度，并导出逐题配对结果及来源表。
- run_local.sh 运行完整 dev/test 的两种方法和评估。

### 运行

在项目根目录创建 Python 3.11 虚拟环境，安装依赖。torch 需要按运行设备安装合适版本。

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install torch
    pip install -r requirements-b.txt -r requirements-generator.txt
    ./run_local.sh

run_local.sh 会优先使用 models/ 内的权重。需要使用其他 BART 时，可通过 EVRAG_GENERATOR_DIR 指定其目录；其他 MiniLM 目录可用以下变量指定。

    export EVRAG_EMBEDDING_MODEL="/absolute/path/to/all-MiniLM-L6-v2"

此 Git 分支包含代码、数据快照和结果。模型需自行获取后放进 models/，或使用环境变量选择本地模型目录。Python 依赖需要安装。桌面的完整本地项目已包含权重。run_local.sh 会重新生成同名 run 和指标文件，重跑前可备份 results。

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

This local project connects Nam's Illgetgood-patch-2 data, Jojo's shared BART generator, Basic RAG, and E's evaluation code. The corpus has 512 passages, each containing at most 150 whitespace-delimited words. The dataset contains 350 questions with 175 each in dev and test.

NLP-only passes the question directly to the fine-tuned BART checkpoint. Basic RAG uses all-MiniLM-L6-v2 embeddings and normalized FAISS IndexFlatIP search to retrieve five candidates. It selects up to four passages in rank order while checking the complete BART input against its 1,024-token budget. Both routes use the same generator and decoding parameters. Empty retrieval produces answer=null and abstained=true without invoking BART. Nonempty retrieval invokes BART without an additional sufficiency gate.

Automatic correctness uses E's token-F1 threshold of 0.5. No human labels were applied. Evidence support uses a lexical heuristic, scored against the contexts actually supplied to BART. Evidence support for NLP-only is N/A. Only the aggregate all group is produced when scenario labels are absent.

All questions have gold answers, so abstention recall is N/A. Three normalized questions occur in both splits. The existing checkpoint may overlap the corpus in its training data. The local checkpoint archive hash differs from the published Release archive hash, so its exact identity with the original Release archive could not be verified. These limitations are recorded in results/basic_v1/run_manifest.json. Parameters are fixed for this draft-data comparison.

Install the dependencies and set EVRAG_GENERATOR_DIR to the extracted checkpoint directory using the commands above. EVRAG_EMBEDDING_MODEL optionally selects a local MiniLM directory. Run ./run_local.sh from this folder. This Git branch includes the input snapshot and results. Download the weights into models/ or set the model environment variables. The launcher uses project-relative paths when local weights are present. Install the Python environment before running.

The results/basic_v1/runs directory contains the four raw run files. metrics_dev.csv and metrics_test.csv contain automatic metrics. paired_comparison.csv aligns both answers for every question and reports individual gold-passage recall and complete-gold retrieval. context_sources.csv links actual generator contexts to the corpus metadata. Source fields are inherited from the supplied data and may not identify a specific original webpage. Empty CSV metrics mean the measure is undefined for these inputs.

The metric named recall_at_k measures whether any gold passage was retrieved. For multi-evidence questions, also inspect gold_fraction_retrieved, all_gold_retrieved, and context_gold_count in paired_comparison.csv. retrieved_docs contains all five candidates, whereas meta.context_doc_ids lists the passages selected for generation.

The local changes add Basic RAG and NLP-only methods, an optional local embedding-path override, progress logging with flushed run output, aggregate evaluation without scenario labels, and paired-output auditing.

实验配置为 configs/basic_rag_v1.yaml，由 EVRAG_CONFIG 选择。共享 default.yaml 保留数据分支原版。
The experiment configuration is configs/basic_rag_v1.yaml, selected with EVRAG_CONFIG.

数据快照来自 Nam / Illgetgood 的 patch-2。basic_metrics.py 基于 Yuxuan / yx-yan 的 e-metrics（PR #29），作为本次实验的独立入口，保留共享 metrics.py。Jojo082099 提供共享 BART 生成器及训练权重。
Data is credited to Nam / Illgetgood. The experiment metric adapter is derived from Yuxuan / yx-yan PR #29. The shared BART generator and checkpoint are credited to Jojo082099.

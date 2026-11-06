> 更新 / Update: MiniLM and live OpenAI validation are now recorded in `docs/reports/b_live_validation.md`. Earlier no-API results below describe the initial delivery.

# Member B — RAG 实现与运行说明 / Implementation & Run Guide

## 0. 本次完成状态 / Delivery status

| 项目 / Item | 状态 / Status |
|---|---|
| A 的数据适配 / Adapt A's data | 完成：1,201 条，dev 953 / test 248 / Complete |
| 共享检索器 / Shared retriever | 完成：Sentence Transformer 接口、FAISS、归一化余弦、向量缓存 / Implemented |
| 共享 LLM 接口 / Shared LLM wrapper | 完成：显式配置的 OpenAI-compatible HTTP 接口 / Implemented |
| Standard RAG / Baseline | 完成：注册、提示词、统一输出 / Implemented |
| 自动化测试 / Automated tests | 32 passed；含真实 FAISS + 可控测试向量 / Includes actual FAISS with deterministic test embeddings |
| 20 条数据集集成检查 / 20-case integration check | 完成：真实 TF-IDF + FAISS 检索、mock 生成、0 次 API 调用 / Completed |
| MiniLM 语义检索验证 / MiniLM semantic retrieval validation | 待完成：依赖可导入，但模型权重下载超时 / Pending: weights download timed out |
| 真实 LLM 问答 / Live LLM generation | 待完成：小组未选择服务和提供 Key / Pending provider and credentials |
| GitHub 推送或 PR / GitHub push or PR | 未完成：连接接口 404，本地 Git 网络不可达 / Not published |

**本次输出不能作为模型效果实验结果。** TF-IDF 是排查流程的词法编码器，不等于 MiniLM；mock 回答是固定标记，不是模型回答。因此不报告答案准确率、拒答效果或正式检索 Recall。

**These outputs are not model-quality results.** TF-IDF is a diagnostic lexical encoder, not MiniLM. The mock generator returns a fixed sentinel, not an LLM answer. No answer accuracy, refusal performance, or formal gold-passage recall is claimed.

## 1. 项目和 B 的贡献 / Project and B's contribution

项目比较三种方法：Standard RAG、相似度阈值拒答、证据充分性判断。B 实现大家共享的检索和生成组件，并将它们连接成 Standard RAG。C/D 在同一基础上实现自己的拒答策略，E 负责正式评估和演示。

The project compares standard RAG, similarity-threshold abstention, and evidence-sufficiency-based abstention. B supplies the shared retrieval/generation backend and the standard baseline. C/D implement their refusal policies, and E owns formal evaluation and the demo.

来源是 GitHub 仓库 `yx-yan/ee6405-evidence-rag` 的 `Illgetgood-patch-1` 分支快照，页面显示提交 `0f1a7906854a5214ba4b53887dcddbf5a25311b5`。这是浏览器下载的源码快照；本地 Git 快照不是远端分支的完整历史，不能直接推送它替代远端历史。

Source: the browser-downloaded `Illgetgood-patch-1` snapshot, shown at commit `0f1a7906854a5214ba4b53887dcddbf5a25311b5`. The locally initialized Git snapshot does not preserve remote ancestry. Apply the patch on a proper clone; do not push the snapshot as replacement repository history.

## 2. 修改清单 / Files and responsibilities

| 文件 / File | 作用 / Purpose |
|---|---|
| `src/evrag/retriever.py` | 读取知识库、编码证据、缓存向量、FAISS Top-k / Corpus loading, evidence embedding, vector cache, FAISS search |
| `src/evrag/llm.py` | `generate(prompt)`、配置检查、HTTP 请求、超时及错误处理 / Configured generation with timeouts and errors |
| `src/evrag/methods/baseline.py` | Standard RAG、提示词、注册 / Standard pipeline, prompt, registration |
| `src/evrag/methods/__init__.py` | 导入并注册 baseline / Register baseline on import |
| `configs/default.yaml` | 缓存路径、检索后端、LLM endpoint/timeout 占位配置 / Minimal shared settings additions |
| `.env.example` | API Key、模型和地址的空白示例 / Empty credential/model/endpoint examples |
| `pyproject.toml` | 增加 NumPy 为核心依赖，保持原 CI 的轻量测试可运行 / Add NumPy for lightweight CI tests |
| `requirements-b.txt` | B 所需依赖安装入口 / B's dependency installation entry point |
| `scripts/b_prepare_data.py` | 从 A 导出生成兼容文件，保留原文与来源 / Explicit data adapter |
| `scripts/b_smoke.py` | dev 集小规模运行，区分 mock/live / Small dev-only integration runs |
| `tests/conftest.py` | 原契约测试的离线依赖替身 / Explicit offline mocks for the original contract test |
| `tests/test_member_b.py` | 排序、缓存、异常、数据映射等测试 / Retrieval, cache, HTTP, adapter tests |
| `data/processed/` | 适配后的派生数据和限制说明 / Derived compatible data and audit |
| `results/b_validation/` | mock 集成结果、版本与测试记录 / Explicitly diagnostic outputs |

没有修改 `schema.py`、`methods/base.py`、E 的 evaluation runner、Streamlit 页面、C/D 模块、A 的原始数据和 notebook。原路径配置经测试确认正确，未修改。

The shared schema/base, E's evaluation runner and demo, C/D modules, and A's source data/notebooks are unchanged. The original project-root path configuration was verified and retained.

## 3. RAG 流程 / How it works

1. **Embedding：**用现成 Sentence Transformer 把文档和问题映射成向量；不训练模型。
2. **归一化：**把每个向量除以自身长度，此时向量内积等于余弦相似度。
3. **FAISS：**用 `IndexFlatIP` 做精确内积检索，默认取 Top-5。它管理搜索，不负责理解文本。
4. **Prompt：**把问题和检索证据发给 LLM。证据以 JSON 形式引用，文档中的文字被视为数据。
5. **Generation：**现成 LLM 返回答案；父类 `Method.run()` 自动记录耗时并构造 `RAGOutput`。

1. **Embedding:** a pretrained Sentence Transformer maps evidence and questions to vectors; no model training.
2. **Normalization:** unit vectors make inner product equal cosine similarity.
3. **FAISS:** exact `IndexFlatIP` search returns the configured Top-k (currently 5).
4. **Prompt:** quoted evidence plus the user question is passed to the generator.
5. **Generation:** the shared LLM returns text; the existing base class supplies timing and `RAGOutput`.

相似度约在 [-1, 1]，是排序分数，不是“答案正确概率”。C 应只在 dev 上调阈值。当前 baseline 没有阈值、judge 或重试；自然语言形式的拒答仍是字符串，按原接口记为 `abstained=False`。调用失败会抛异常，不会伪装成模型拒答。

Cosine similarity is a ranking score, not a correctness probability. C should tune thresholds on dev only. The baseline adds no threshold, judge, or retry. Natural-language refusals remain string answers (`abstained=False`) under the current contract. API failures raise exceptions instead of being counted as refusals.

## 4. 数据适配与限制 / Data adaptation and limitations

- `data/mts_dialog/corpus.jsonl` 的 `id/context` 映射成 `doc_id/text`；只索引 context。
- 从 `final_annotated_dataset.csv` 补回精简表中缺失的 `question`；`gold_answer` 转为 JSON 数组 `gold_answers`。
- 不将问题、标准答案、dialogue 或 scenario 放进向量索引。
- 保留临床描述 `scenario`，另加 `evidence_scenario_status=unlabelled`。
- `source_doc_id` 仅表示来源对应关系；不是经过人工确认的 gold passage，因此 `gold_doc_ids=[]`。
- A 的原始文件不改；脚本拒绝覆盖没有 B 审计标记的已有 processed 数据。

The adapter maps `id/context` to `doc_id/text`, recovers questions from the full annotated CSV, and serializes reference answers as a JSON list. It embeds only context text. Encounter descriptions are preserved instead of relabelled as evidence categories. Source IDs are provenance only, not verified support labels. Existing teammate-produced processed files are protected against accidental overwrite.

**需要 A/E 后续确认：**

- 标准答案基于 dialogue 和 context 生成，但语料只含 context。例：哮喘样例问“恢复吸入器后的改善和医生建议”，对应 context 只有 `PASTMEDICALHX: Asthma.`，不足以支持完整答案。
- `scenario` 目前不是原设计的“证据充分/缺失/干扰”标签，不能据此计算拒答指标。
- 许多问题使用 “the patient”，全局检索可能找到另一个病例。需要明确任务是否为全局病例检索，以及问题是否包含足够的病例识别信息。
- 对无答案样本，需要检查整个检索库，而不仅是对应原文。正式 gold passage、拒答标签和 dev/test 去重由 A/E 审核。

**A/E follow-up:** reference answers can contain dialogue-only facts, current scenarios are encounter descriptions, and globally searching ambiguous patient questions can mix encounters. Before formal comparison, audit evidence sufficiency, corpus-wide unanswerability, source identity, and split overlap. B does not invent these labels.

## 5. 安装和运行 / Setup and commands

推荐独立 Python 3.11 或 3.12 环境。在仓库根目录执行：
Use a dedicated Python 3.11/3.12 environment and run from the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-b.txt
python scripts/b_prepare_data.py
python -m pytest -q
```

本次本机运行环境复用了已有 Python 3.12 环境中的依赖，并在项目独立 `.venv` 中补齐所缺包。这个本机环境链接不随 ZIP 分发；其他机器按以上命令重新安装。准确版本在 `results/b_validation/environment.json`。

This session reused read-only dependencies from an existing local Python 3.12 environment and installed missing packages into the project's own virtual environment. That machine-specific environment is not distributed. Install normally on another machine; tested versions are recorded in `environment.json`.

**无需 API 的语义检索 + mock 生成：**首次需要联网下载 MiniLM（本次下载未成功）。
**Semantic retrieval with mock generation:** first download requires model-network access; it did not complete in this session.

```bash
python scripts/b_smoke.py --mock-llm --limit 20
```

已在别处下载完整的 Sentence Transformer 模型目录时，可指定本地路径：
If a full Sentence Transformer model directory is already available:

```bash
python scripts/b_smoke.py --mock-llm --limit 20 --embedding-model /path/to/all-MiniLM-L6-v2
```

**本次实际跑过的纯本地诊断：**
**Offline diagnostic actually executed in this session:**

```bash
python scripts/b_smoke.py --mock-llm --offline-tfidf --limit 20
```

此命令使用 scikit-learn 的 TF-IDF（Sentence Transformers 的依赖）及真实 FAISS，不需要模型权重或 API。输出在 `results/b_validation/`，每条记录都写有 `generation_mode=mock`、`embedding_model=diagnostic:tfidf_word_1_2`、`evaluation_eligible=false`。不要将它混入正式 `results/runs/`。

This uses actual TF-IDF and FAISS, with no weights or API. Every output is marked as mock, diagnostic TF-IDF, and ineligible for formal evaluation. Outputs stay outside E's formal `results/runs/` directory.

## 6. 配置真实 LLM / Configure the real LLM later

小组统一选择一个支持 OpenAI-compatible `/chat/completions` 的服务后，把 `.env.example` 复制成 `.env`，在本机填写：
Once the team selects a compatible provider, copy `.env.example` to `.env` and fill locally:

```dotenv
LLM_API_KEY=your-key-on-your-computer
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=the-team-agreed-model
```

`.env` 已被 gitignore 排除，不要上传 Key。这里的地址和模型是占位示例，不是可用服务。温度为 0、max_tokens 为 256，统一配置在 YAML；这个封装没有自动付费重试。尚未实际验证任何服务商的兼容性。

The URL/model above are placeholders. Keep keys out of Git and chat. Shared YAML sets temperature 0 and max_tokens 256. There are no hidden paid retries. No provider compatibility was live-tested yet.

然后运行（该步骤会调用 API）：
Then run (this step makes paid/provider-account API calls):

```bash
python scripts/b_smoke.py --limit 20
# 或使用 E 的原有入口 / or E's existing runner:
python -m evrag.evaluation.run_experiment --method baseline --split dev --limit 20
```

## 7. C/D 如何复用 / Shared API for C and D

```python
from evrag import retriever, llm

# Requires downloaded MiniLM weights, but no paid API.
docs = retriever.search("What medications does the patient take?", top_k=5)
for doc in docs:
    print(doc.doc_id, doc.score, doc.text)

# Requires configured LLM credentials and endpoint.
answer = llm.generate("Your evidence-grounded prompt")
```

```python
from evrag.methods import get_method
out = get_method("baseline").run("Your question", question_id="q1")
print(out.to_dict())
```

默认检索器在一个 Python 进程中复用模型、文档及索引。更换语料/配置后重启进程，或显式调用 `retriever.get_retriever.cache_clear()`。磁盘缓存以模型名称和文档内容散列命名；换语料后重建向量缓存。FAISS 索引从缓存向量快速重建。

The default retriever is reused within a process. Restart or clear the cached instance after data/config changes. Disk-cached vectors are keyed by the model name and corpus content; FAISS is rebuilt from those vectors. Custom test encoders disable persistent caching to avoid misleading cache reuse.

## 8. 测试解释 / What the tests establish

- 余弦排序、Top-k 上限和合法性：用可控向量验证数学结果；FAISS 和 NumPy 两种精确搜索一致。
- 实际编码输入只含证据文本，避免 query/gold answer 泄漏。
- 相同语料复用缓存；语料内容变化会使缓存失效。
- LLM HTTP 请求内容、超时、缺 Key、错误响应和 429 等异常通过替身传输测试。
- baseline 注册、提示词、耗时和统一输出满足契约；原契约测试使用明确的离线替身，不访问网络。
- 数据适配保留源文件，不编造证据标签，并保护已有 processed 数据。

Tests establish exact cosine math, FAISS behavior, index-input isolation, cache invalidation, HTTP payload/error semantics, baseline registration and schema compliance, and safe data adaptation. Mock tests validate plumbing, not pretrained-model quality or live-provider behavior.

## 9. 如何提交和说明贡献 / Applying and explaining the contribution

初始交付后，已通过网页上传创建草稿 PR #17（B 分支到 A 的数据分支）。本轮真实验证先保留在本地，尚未上传；旧交付 ZIP 不包含本轮更新。

Draft PR #17 was subsequently created through GitHub web upload. This round of live validation remains local and is not included in the original delivery ZIP.

在可访问仓库的正常 clone 中，基于含 A 数据的分支创建 B 分支，然后先检查再应用：
In a proper clone containing A's branch/data, create your B branch, then check and apply:

```bash
git switch -c b-standard-rag
git apply --check /path/to/member_b_changes.patch
git apply /path/to/member_b_changes.patch
python -m pip install -r requirements-b.txt
python -m pytest -q
```

若 A/E 更新过相同文件，先人工合并差异；不要强制覆盖。保留本次对话的分享链接，按项目要求披露 AI 辅助，并理解代码后使用你自己的账号提交。

Resolve overlapping teammate changes manually. Do not overwrite remote history. Retain the shareable chat link for the team's AI disclosure and commit through your own account after reviewing and understanding the implementation.

**可以准确表述的贡献：**我实现了共享的检索/生成接口和 Standard RAG，完成了数据格式适配、自动化测试及离线集成检查。最新 MiniLM 与真实 LLM 验证结果见 docs/reports/b_live_validation.md；正式评测仍依赖经审核的数据。

**Accurate contribution statement:** I implemented the shared retrieval/generation APIs and Standard RAG baseline, adapted the data format, and completed automated tests plus an offline integration check. See docs/reports/b_live_validation.md for current MiniLM and live LLM results; formal evaluation still requires audited evidence data.

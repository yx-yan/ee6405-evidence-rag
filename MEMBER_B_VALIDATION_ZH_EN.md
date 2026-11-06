# Member B 实际验证说明 / Live validation report

Date: 2026-11-03. This report supersedes the initial no-API status.

## 1. 本轮做了什么 / What was completed

- 保留共享接口 `retriever.search()`、`llm.generate()` 和 `baseline`，不修改 E 的输出契约。
  Implemented shared retrieval/generation interfaces and Standard RAG using E's existing contract.
- 从官方 Hugging Face 下载真实 MiniLM 权重及分词器；此前是下载连接超时，不是模型不能用。本轮下载成功，无需用户手动下载。
  Downloaded actual MiniLM weights/tokenizer from the official Hugging Face repository. The earlier download timeout is resolved.
- 对 1,201 篇 context 建立 1,201 × 384 的归一化向量与 FAISS IndexFlatIP，并验证缓存复用。
  Built normalized 1,201 × 384 document vectors, an exact FAISS index, and verified embedding-cache reuse.
- 20 条 dev 查询的 FAISS 分数与独立 NumPy 余弦参考计算一致。
  All 20 dev queries matched the independent NumPy cosine reference scores.
- 使用 OpenAI `gpt-4.1-mini-2025-04-14` 完成 20 条真实端到端调用；全部返回有效回答，finish_reason 均为 stop。
  Completed 20 live end-to-end dev cases with OpenAI; all returned valid answers with finish_reason=stop.
- 自动化测试：32 passed。测试不读取本地真实 Key，避免测试误用真实账户。
  Automated checks: 32 passed. Offline LLM tests do not load the real local credential.

模型选择是本轮 B 的试跑配置；C/D 正式比较前仍需统一模型、top_k、提示词和数据版本。
The model is B's pilot configuration. C/D comparisons still require a common model, top_k, prompt policy and frozen dataset.

## 2. 实际 API 用量 / Actual API usage

| Item | Value |
|---|---:|
| Successful generation requests / 成功生成请求 | 20 |
| Input tokens / 输入 | 9,721 |
| Output tokens / 输出 | 854 |
| Total tokens / 合计 | 10,575 |
| Truncated completions / 长度截断 | 0 |
| Automatic paid retries / 自动付费重试 | 0 |

另有一次认证 models 列表查询，不属于生成请求。金额以 OpenAI 账单为准，此报告不估算未经核实的费用。
One authenticated model-list request was also made. Billing should be checked in OpenAI's dashboard; this report does not infer charges from unverified prices.

## 3. A 的数据：可接入，但仍需审核 / A's data: usable for integration, pending audit

本次审核基于先前从 A 的 `Illgetgood-patch-1` 分支下载的本地快照，不代表重新读取了当前 GitHub 最新版本。
This review uses the local snapshot previously downloaded from A's branch, not a fresh audit of the current remote state.

| Item | Finding |
|---|---|
| QA / corpus | 1,201 / 1,201 |
| dev / test | 953 / 248 |
| Duplicate question IDs / 重复题号 | 0 |
| Repeated context occurrences / 重复 context 次数 | 114 |
| Identical contexts present in both splits / 跨 split 相同 context 组 | 20 |
| Contexts under 10 words / 少于 10 词 | 447 |
| Audited sufficient/missing/distracting labels / 三类证据标签 | Not available / 尚无 |

短文本和重复文本是审核线索，并非自动判错；跨 split 相同 context 是否违反拆分原则，需要 A/E 确认。
Short/duplicate contexts are audit flags, not automatic errors. A/E should verify whether cross-split duplicates violate the agreed split policy.

明确例子：`QA_MED_EF2D127DE37B942B` 的问题及 gold 涉及恢复吸入器后的改善和随身携带建议，但 context 只有 `PASTMEDICALHX / Asthma.`。因此该来源文档本身不能支持完整答案。
Concrete example: this item's gold describes improvement after resuming an inhaler and advice to carry it, while its source context only states a history of asthma. That document cannot support the full answer.

A 需要确认：
1. 证据究竟采用 context、dialogue，还是经过审核的组合；保证 gold 与可检索证据一致。
2. 按项目要求构造并审核证据充分、缺失、干扰标签，尤其检查“缺失”是否对整个 corpus 成立。
3. 明确不同患者记录在全局检索时的身份区分方式，审核重复与 dev/test 拆分，冻结数据版本。

A should confirm the allowed evidence source, align gold answers with retrievable evidence, audit all three evidence scenarios, address patient identity and split policy, and freeze a data version.

来源文档在 top-5 中出现 4/20，仅用于追踪来源，不是 gold recall@5；没有经过审核的 gold_doc_ids，不能据此报告正式检索召回率。
The source document appeared in top-5 for 4/20 cases. This is provenance tracking, not gold recall@5: verified supporting document IDs are still missing.

## 4. 文件和重跑 / Files and reproduction

- `results/runs/baseline_dev20.jsonl`: 20 条真实回答、检索文档、相似度、耗时和 API usage / live answers, evidence, scores, latency and usage.
- `results/runs/b_smoke_summary.json`: 完整本轮汇总 / full live-run summary.
- `results/b_validation/live_usage_summary.json`: 简要用量 / compact token accounting.
- `results/b_validation/minilm_retrieval_validation.json`: 真实向量验证 / dense retrieval verification.
- `results/b_validation/a_source_review.json`: A 数据审核线索与具体例子 / data review findings and example.
- `scripts/b_download_model.py`: 官方固定版本下载 / pinned official-model download.
- `scripts/b_validate_retrieval.py`: 不调用 API 的真实检索验证 / real retrieval verification without API calls.
- `scripts/b_smoke.py`: dev 集成检查；不自动重试付费请求，逐条保存完成结果 / dev integration with incremental saving and no paid retries.

```bash
python -m pip install -r requirements-b.txt
python scripts/b_download_model.py
python -m pytest -q
python scripts/b_validate_retrieval.py
python scripts/b_smoke.py --mock-llm --embedding-model models/all-MiniLM-L6-v2
# Paid: requires your own .env settings (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL)
python scripts/b_smoke.py --embedding-model models/all-MiniLM-L6-v2 --limit 20
```

本机使用项目 `.venv/bin/python`。本机 `.env` 权限为 600 且被 git 忽略；交付包不含 Key、虚拟环境、模型权重或向量缓存。下载脚本可恢复权重。相同输出文件名会被新一次运行覆盖，重跑前请保留需要的结果。
On this machine use `.venv/bin/python`. The local credential file is mode 600 and git-ignored. The bundle excludes credentials, environments, model weights and caches. Download weights with the supplied script. Repeated runs overwrite matching result filenames; preserve earlier results first.

共享接口默认按官方模型名加载；重跑脚本的 `--embedding-model` 指向本地权重。C/D 若要离线使用共享入口，应将自己的运行配置 embedding_model 设为该本地目录。
The shared default loads the official model ID; the scripts explicitly select local weights. For offline C/D use, set their runtime embedding_model to the local model directory.

## 5. B 还需要做什么 / B's remaining responsibilities

当前完成的是实现、单元测试、真实检索检查和 20 条真实试跑，不代表答案正确率验证完成。
Implementation, unit checks, actual retrieval verification and a 20-case live pilot are complete; answer quality is not yet validated.

待 A/E 确认证据数据、gold 支持文档及统一模型后，B 需要计算有定义的 recall@k，运行完整 dev/test baseline，给 C/D 提供共享接口使用说明，并整理 2–3 个经过人工核对的失败案例。不要把当前 context-only 试跑当作最终方法对比。
Once A/E confirm evidence, gold support IDs and the shared model, B should compute valid recall@k, run the full dev/test baseline, support C/D integration and document 2–3 manually checked failure cases. Current context-only pilot outputs are not final comparative results.

完整试跑日志保存在本地。此处分享验证摘要与复现代码。
Full pilot logs are stored locally. This update shares the validation summaries and reproduction code.

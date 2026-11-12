"""Audit completed run files and produce a paired, passage-level comparison."""

import csv
import json
from pathlib import Path

from evrag.evaluation.metrics import score_rows, load_dataset, load_runs
from evrag.config import ROOT, RESULTS_DIR, load_config

rows = list(csv.DictReader((ROOT / load_config()["data"]["dataset"]).open()))
questions = {r["question_id"]: r for r in rows}
corpus = {d["doc_id"]: d for d in map(json.loads, (ROOT / load_config()["data"]["corpus"]).open())}
results = RESULTS_DIR
comparison = []
sources = []
for split in ("dev", "test"):
    dataset = load_dataset(ROOT / load_config()["data"]["dataset"], split)
    outputs = {}
    for method in ("nlp_only", "basic_rag"):
        runs = load_runs([results / "runs" / f"{method}_{split}.jsonl"])
        assert len(runs) == len(dataset)
        assert len({r["question_id"] for r in runs}) == len(runs)
        assert {r["question_id"] for r in runs} == set(dataset)
        outputs[method] = {r["question_id"]: r for r in runs}
    scores = {
        method: {r["question_id"]: r for r in score_rows(list(runs.values()), dataset)}
        for method, runs in outputs.items()
    }
    for qid, q in dataset.items():
        nlp, rag = outputs["nlp_only"][qid], outputs["basic_rag"][qid]
        context_ids = rag["meta"]["context_doc_ids"]
        assert len(context_ids) <= 4
        assert rag["meta"]["input_tokens"] <= rag["meta"]["max_source_tokens"]
        assert set(context_ids) <= {d["doc_id"] for d in rag["retrieved_docs"]}
        for d in rag["retrieved_docs"]:
            assert d["text"] == corpus[d["doc_id"]]["text"].strip()
        gold = set(q["gold_doc_ids"])
        retrieved = {d["doc_id"] for d in rag["retrieved_docs"]}
        context = set(context_ids)
        comparison.append({
            "split": split, "question_id": qid, "question": q["question"],
            "gold_answers": json.dumps(q["gold_answers"]), "gold_doc_ids": json.dumps(q["gold_doc_ids"]),
            "nlp_answer": nlp["answer"], "rag_answer": rag["answer"],
            "nlp_token_f1": scores["nlp_only"][qid]["f1"],
            "rag_token_f1": scores["basic_rag"][qid]["f1"],
            "nlp_rouge_l": scores["nlp_only"][qid]["rouge_l"],
            "rag_rouge_l": scores["basic_rag"][qid]["rouge_l"],
            "rag_context_doc_ids": json.dumps(context_ids),
            "gold_count": len(gold), "retrieved_gold_count": len(gold & retrieved),
            "context_gold_count": len(gold & context),
            "gold_fraction_retrieved": len(gold & retrieved) / len(gold) if gold else None,
            "all_gold_retrieved": gold <= retrieved if gold else None,
            "nlp_latency_ms": nlp["latency_ms"], "rag_latency_ms": rag["latency_ms"],
        })
        for doc_id in context_ids:
            d = corpus[doc_id]
            sources.append({
                "question_id": qid, "doc_id": doc_id,
                **{k: d.get(k, "") for k in ("article_id", "title", "section", "disease", "source_url", "text")},
            })

for name, entries in [("paired_comparison.csv", comparison), ("context_sources.csv", sources)]:
    with (results / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
print(f"Verified {len(comparison)} paired questions and {len(sources)} evidence links.")

lines = [
    "# Basic RAG v1 results / 结果",
    "",
    "Fixed-configuration comparison on A's WikiDoc data (data/processed).",
    "A 的 WikiDoc 数据（data/processed）上的固定配置对比。两种方法共生成 700 条输出，配对覆盖 350 道题。",
    "",
    "| Split | Method | Questions | Token F1 | ROUGE-L | Automatic correctness | Retrieval hit | Mean latency ms |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for split in ("dev", "test"):
    for metric in csv.DictReader((results / f"metrics_{split}.csv").open()):
        values = [split, metric["method"], metric["n"], metric["token_f1"], metric["rouge_l"],
                  metric["answer_correctness"], metric["recall_at_k"] or "N/A", metric["latency_mean_ms"]]
        lines.append("| " + " | ".join(values) + " |")
lines += [
    "",
    "Automatic correctness uses token F1 >= 0.5. These values have not been manually adjudicated.",
    "自动正确率使用 Token F1 >= 0.5，尚未人工判分。",
    "",
    "The checkpoint may overlap the corpus in training, and three normalized questions occur in both splits.",
    "现有权重可能在训练时见过语料，dev/test 有三个归一化后的重复问题。",
    "",
    "All questions have gold answers, so abstention recall is N/A. Evidence-support heuristics cannot establish factual correctness.",
    "所有题目都有参考答案，拒答召回率为 N/A。词汇证据支持启发式不能证明事实正确。",
    "",
    "Latency includes first-call model/index loading and reflects CPU runs on this machine.",
    "延迟包含各次运行首题的模型/索引加载，反映当前电脑 CPU 运行情况。",
    "",
    "Read docs/reports/basic_rag_v1.md for setup and source provenance.",
    "环境设置及来源追溯见 docs/reports/basic_rag_v1.md。",
]
(results / "RESULTS_ZH_EN.md").write_text("\n".join(lines) + "\n")

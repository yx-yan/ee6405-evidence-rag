"""D's offline tests for the LiveQA-Med converter: parsing, judged-answer mapping and splits."""
from __future__ import annotations
import csv
import json
from evrag.data import liveqa

XML = """<?xml version="1.0" encoding="UTF-8"?><NLM-QUESTIONS>
<NLM-QUESTION qid="TQ1"><Original-Question qfile="1.txt"><SUBJECT>PKD</SUBJECT>
<MESSAGE>how many   people have polycystic kidney disease?</MESSAGE></Original-Question>
<NLM-Summary>How common is PKD?</NLM-Summary>
<ANNOTATIONS><FOCUS fid="F1" fcategory="Problem">polycystic kidney disease</FOCUS><TYPE tid="T1" hasFocus="F1">FREQUENCY</TYPE></ANNOTATIONS>
<ReferenceAnswers><RefAnswer aid="TQ1A1"><ANSWER>About 1 in 500 people.</ANSWER><AnswerURL>u</AnswerURL></RefAnswer></ReferenceAnswers>
</NLM-QUESTION>
<NLM-QUESTION qid="TQ2"><Original-Question qfile="2.txt"><SUBJECT>drug</SUBJECT><MESSAGE>is zolmitriptan gluten free?</MESSAGE></Original-Question>
<NLM-Summary>Does zolmitriptan contain gluten?</NLM-Summary><ANNOTATIONS/>
<ReferenceAnswers><RefAnswer aid="TQ2A1"><ANSWER>No gluten is listed.</ANSWER></RefAnswer></ReferenceAnswers>
</NLM-QUESTION></NLM-QUESTIONS>"""
QRELS = """1 4-Excellent GHR_0000804_Sec2.txt
1 1-Incorrect GHR_0000804_Sec1.txt
1 3-Incomplete ADAM_0000001_Sec1.txt
2 4-Excellent MPlusDrugs_0000002_Sec1.txt
"""
ANSWERS = ('AnswerID,Answer\n'
           'GHR_0000804_Sec2.txt,"Question: How many people are affected by PKD ?\nURL: x\nAnswer: Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people."\n'
           'GHR_0000804_Sec1.txt,"Question: What is PKD ?\nAnswer: PKD is a disorder."\n'
           'ADAM_0000001_Sec1.txt,"Question: x\nAnswer: "\n'
           'MPlusDrugs_0000002_Sec1.txt,"Question: x\nAnswer: "\n')
CORPUS = [{"doc_id": "GHR_0000804-2#0", "text": "Polycystic kidney disease — Frequency\n\nPolycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States."},
          {"doc_id": "GHR_0000804-2#1", "text": "Polycystic kidney disease — Frequency\n\nThe autosomal dominant form is much more common."},
          {"doc_id": "GHR_0000804-1#0", "text": "Polycystic kidney disease — Overview\n\nPKD is a disorder."}]


def test_build(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    (src / "test.xml").write_text(XML, encoding="utf-8")
    (src / "All-qrels_x.txt").write_text(QRELS, encoding="utf-8")
    (src / "All-2479-Answers-x.csv").write_text(ANSWERS, encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(d) + "\n" for d in CORPUS), encoding="utf-8")
    manifest = liveqa.build(src, tmp_path / "out", corpus, seed=3)
    rows = list(csv.DictReader((tmp_path / "out" / "dataset.csv").open(encoding="utf-8", newline="")))
    assert list(rows[0]) == list(liveqa.COLUMNS) and len(rows) == 2
    by_id = {r["question_id"]: r for r in rows}
    pkd, drug = by_id["LQ_TQ1"], by_id["LQ_TQ2"]
    assert pkd["question"] == "PKD\nhow many people have polycystic kidney disease?"
    assert json.loads(pkd["gold_answers"]) == ["About 1 in 500 people."]
    assert json.loads(pkd["gold_doc_ids"]) == ["GHR_0000804-2#0", "GHR_0000804-2#1"]   # 4-Excellent, both chunks
    assert pkd["evidence_scenario_status"] == "judged" and pkd["focus"] == "polycystic kidney disease"
    assert pkd["qtype"] == "FREQUENCY" and pkd["summary"] == "How common is PKD?" and pkd["scenario"] == "R"
    assert json.loads(drug["gold_doc_ids"]) == [] and drug["evidence_scenario_status"] == "reference_only"
    assert manifest["counts"]["good_answers_in_corpus"] == 1 and manifest["counts"]["good_answers_outside_corpus"] == 2
    assert {r["split"] for r in rows} <= {"dev", "test"}
    assert liveqa.medquad_pair_id("MPlusHealthTopics_0000585_Sec1.txt") == "MPlus_Health_Topics_0000585-1"
    assert liveqa.medquad_pair_id("CancerGov_0000007_1_Sec2.txt") == "CancerGov_0000007_1-2"
    assert liveqa.medquad_pair_id("ADAM_0000001_Sec1.txt") is None

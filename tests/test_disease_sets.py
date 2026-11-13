"""Offline tests for the coronavirus / cancer / neuro set builders."""
from __future__ import annotations

import json

from evrag.data import disease_sets, wikidoc

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000001" source="CancerGov" url="https://www.cancer.gov/x">
<Focus>Skin Cancer</Focus>
<QAPairs>
  <QAPair pid="1"><Question qid="0000001-1" qtype="information">What is (are) Skin Cancer ?</Question>
    <Answer>Skin cancer is a disease.</Answer></QAPair>
  <QAPair pid="2"><Question qid="0000001-2" qtype="research">What research is being done?</Question>
    <Answer>NCI supports research.</Answer></QAPair>
  <QAPair pid="3"><Question qid="0000001-3" qtype="treatment">What are the treatments for Skin Cancer ?</Question>
    <Answer>{long}</Answer></QAPair>
</QAPairs>
</Document>"""


def test_chunk_spans_match_chunk_words():
    words = [str(i) for i in range(400)]
    chunks = wikidoc.chunk_words(" ".join(words))
    assert [" ".join(words[s:e]) for s, e in disease_sets.chunk_spans(len(words))] == chunks
    assert disease_sets.chunk_spans(10) == [(0, 10)]


def test_answer_word_span_and_gold_chunks():
    context = "alpha beta gamma delta"
    assert disease_sets.answer_word_span(context, context.index("beta"), "beta gamma") == (1, 3)
    assert disease_sets.answer_word_span(context, context.index("eta"), "eta") == (1, 2)
    spans = [(0, 150), (120, 270), (240, 300)]
    assert disease_sets.gold_chunks(spans, 125, 140) == [0, 1]
    assert disease_sets.gold_chunks(spans, 100, 200) == [0, 1]


def test_merge_and_sample_merges_repeats():
    candidates = [
        {"question": "What is X?", "answer": "a", "doc_ids": ["d0"]},
        {"question": "what is x", "answer": "b", "doc_ids": ["d1", "d0"]},
        {"question": "What is Y?", "answer": "c", "doc_ids": ["d2"]},
    ]
    rows = disease_sets.merge_and_sample(candidates, "t")
    assert [r["question_id"] for r in rows] == ["t_Q0", "t_Q1"]
    assert json.loads(rows[0]["gold_answers"]) == ["a", "b"]
    assert json.loads(rows[0]["gold_doc_ids"]) == ["d0", "d1"]
    assert [r["split"] for r in rows] == ["dev", "test"]
    assert len(disease_sets.merge_and_sample(candidates, "t", limit=1)) == 1


def test_medquad_records_skips_research_and_long_answers(tmp_path):
    (tmp_path / "0000001.xml").write_text(PAGE.format(long=" ".join(["w"] * 400)), encoding="utf-8")
    corpus, questions = disease_sets.medquad_records(tmp_path, "can", "Cancer")
    assert [q["question"] for q in questions] == ["What is (are) Skin Cancer ?"]
    assert len(corpus) == 1 + 1 + 4          # every answer is in the corpus, the long one in 4 windows
    assert corpus[0]["text"] == "Skin Cancer: Skin cancer is a disease."
    assert corpus[2]["section"] == "treatment" and corpus[2]["source_url"] == "https://www.cancer.gov/x"


def test_covid_qa_records_keeps_coronavirus_papers():
    def paper(doc_id, title, body):
        context = f"{title}\n\nhttps://example.org/{doc_id}\n\n{body}"
        return {"paragraphs": [{"document_id": doc_id, "context": context, "qas": [
            {"question": f"Q{doc_id}?", "answers": [{"text": "answer here",
                                                     "answer_start": context.index("answer here")}]}]}]}
    squad = {"data": [paper(1, "SARS coronavirus spike", "x " * 200 + "answer here"),
                      paper(2, "HIV in children", "answer here")]}
    corpus, questions = disease_sets.covid_qa_records(squad)
    assert {d["article_id"] for d in corpus} == {"cov_paper_1"}
    assert corpus[0]["source_url"] == "https://example.org/1"
    gold = json.loads(questions[0]["gold_doc_ids"])
    assert all("answer here" in next(d["text"] for d in corpus if d["doc_id"] == g) for g in gold)

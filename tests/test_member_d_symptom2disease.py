"""D's offline tests for the Symptom2Disease converter: sampling, page mapping and gold chunks."""
from __future__ import annotations
import csv
import json
import pytest
from evrag.data import symptom2disease as s2d

CORPUS = [{"doc_id": "MP_1-1#0", "text": "Acne — Overview\n\nAcne is a skin condition."},
          {"doc_id": "NIDDK_2-3#0", "text": "Asthma — Symptoms\n\nWheezing and coughing."},
          {"doc_id": "NIDDK_2-1#0", "text": "Asthma — Overview\n\nAsthma is a lung disease."},
          {"doc_id": "MP_3-1#0", "text": "Migraine — Overview\n\nA migraine is a headache."}]


def test_build(tmp_path):
    src = tmp_path / "Symptom2Disease.csv"
    rows = ["," + "label,text"]
    rows = [",label,text"]
    k = 0
    for label, n in (("Acne", 4), ("Bronchial Asthma", 3), ("Cervical spondylosis", 2), ("Migraine", 3)):
        for j in range(n):
            rows.append(f'{k},{label},"Description {j} of {label} with itching and pain."')
            k += 1
    rows.append(f'{k},Acne,"Description 0 of Acne with itching and pain."')   # duplicate text: dropped
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(d) + "\n" for d in CORPUS), encoding="utf-8")
    manifest = s2d.build(src, tmp_path / "out", corpus, per_label=2, test_fraction=0.5, seed=1)
    items = list(csv.DictReader((tmp_path / "out" / "dataset.csv").open(encoding="utf-8", newline="")))
    assert list(items[0]) == list(s2d.COLUMNS) and len(items) == 8     # 2 per label
    assert manifest["counts"]["distinct_texts"] == 12 and manifest["counts"]["status"] == {"mapped": 6, "no_page": 2}
    by_label = {}
    for r in items:
        by_label.setdefault(r["label"], []).append(r)
    asthma = by_label["Bronchial Asthma"][0]
    assert json.loads(asthma["gold_doc_ids"]) == ["NIDDK_2-3#0"] and asthma["focus"] == "Asthma"   # Symptoms chunk only
    acne = by_label["Acne"][0]
    assert json.loads(acne["gold_doc_ids"]) == ["MP_1-1#0"] and json.loads(acne["gold_answers"]) == ["Acne"]
    spondylosis = by_label["Cervical spondylosis"][0]
    assert spondylosis["evidence_scenario_status"] == "no_page" and spondylosis["focus"] == "" and spondylosis["gold_doc_ids"] == "[]"
    for label, group in by_label.items():
        assert sorted(r["split"] for r in group) == ["dev", "test"]
    assert all(r["scenario"] == "S" for r in items)


def test_unknown_label_rejected(tmp_path):
    src = tmp_path / "s.csv"
    src.write_text(',label,text\n0,Mystery,"text here"\n', encoding="utf-8")
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps(CORPUS[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        s2d.build(src, tmp_path / "out", corpus)

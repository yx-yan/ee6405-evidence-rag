"""Offline tests for the shared generator's input format and its training-data builder."""
from __future__ import annotations
import json
import pytest
from evrag import generator
from evrag.data import generator_data as gd
from evrag.schema import RetrievedDoc


def test_format_input():
    assert generator.format_input("  What is   asthma? ") == "question: What is asthma?"
    assert generator.format_input("What is asthma?", []) == "question: What is asthma?"
    docs = [RetrievedDoc("d1", .9, "Asthma — Overview\n\nAsthma is a lung disease."), "  Wheezing is common. "]
    assert generator.format_input("What is asthma?", docs) == (
        "question: What is asthma? context: [1] Asthma — Overview Asthma is a lung disease. [2] Wheezing is common.")
    with pytest.raises(ValueError):
        generator.format_input("   ")


def test_settings_and_decoding(monkeypatch, tmp_path):
    cfg = {"generator": {"model_dir": "TODO", "num_beams": 1, "max_new_tokens": 50}}
    assert generator.settings(cfg)["model_dir"] is None
    monkeypatch.setenv("EVRAG_GENERATOR_DIR", str(tmp_path))
    assert generator.settings(cfg)["model_dir"] == str(tmp_path)
    assert generator.decoding(cfg["generator"]) == {"num_beams": 1, "max_new_tokens": 50, "no_repeat_ngram_size": 3}
    assert generator.decoding({"num_beams": 4})["early_stopping"] is True


def test_download_from_release(monkeypatch, tmp_path):
    import hashlib
    import shutil
    import zipfile
    archive = tmp_path / "w.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("config.json", "{}")
    release = {"repo": "o/r", "tag": "t", "asset": "w.zip",
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}

    def fake_gh(cmd, check):   # stands in for `gh release download … -D <dir>`
        shutil.copy(archive, cmd[cmd.index("-D") + 1])

    monkeypatch.setattr(generator.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="releases/tag/t"):
        generator.download(str(tmp_path / "m1"), release)
    monkeypatch.setattr(generator.shutil, "which", lambda _: "gh")
    monkeypatch.setattr(generator.subprocess, "run", fake_gh)
    generator.download(str(tmp_path / "m1"), release)
    assert (tmp_path / "m1" / "config.json").read_text() == "{}"
    with pytest.raises(RuntimeError, match="SHA-256"):
        generator.download(str(tmp_path / "m2"), {**release, "sha256": "0" * 64})
    assert not (tmp_path / "m2").exists()


def test_rouge_l():
    assert generator.rouge_l("dry itchy skin", "Dry itchy skin") == 1.0
    assert generator.rouge_l("", "x") == 0.0
    assert 0 < generator.rouge_l("itchy red skin patches", "dry itchy skin") < 1


def test_strip_webmd_script_and_chunks():
    js = ("var s_context; s_context= s_context || {}; if(webmd.useragent){ $(function() { x(); }); } "
          "else { $(function(){ $('.c').remove(); }); } Gastritis is an inflammation. It hurts.")
    assert gd.strip_webmd_script(js) == "Gastritis is an inflammation. It hurts."
    assert gd.strip_webmd_script("Plain text. More.") == "Plain text. More."
    text = " ".join(f"Sentence {i} has five words." for i in range(10))
    spans = gd.chunk_spans(text, max_words=12)
    assert all(len(text[a:b].split()) <= 12 for a, b in spans)
    assert " ".join(text[a:b] for a, b in spans) == text


def test_split_of_is_deterministic():
    assert gd.split_of("What is asthma?") == gd.split_of("what is ASTHMA")
    splits = [gd.split_of(f"question number {i}") for i in range(2000)]
    assert 0.85 < splits.count("train") / 2000 < 0.95


def test_assemble_dedupes_and_keeps_held_out_clean():
    rows = [gd.example("s", "closed", "train", "What causes gout?", "Uric acid crystals build up in the joint."),
            gd.example("s", "closed", "train", "What causes gout?", "Uric acid crystals build up in the joint."),
            gd.example("s", "closed", "val", "What is acne?", "A skin condition with pimples and blackheads."),
            gd.example("t", "closed", "train", "what is ACNE", "Acne is a common skin problem of teenagers."),
            gd.example("s", "closed", "train", "Is it bad?", "No."),
            gd.example("s", "closed", "train", "What is eczema?", "Eczema makes skin dry, red and itchy.")]
    splits, dropped = gd.assemble(rows, exclude={gd.norm("What is eczema?")})
    assert [e["question"] for e in splits["train"]] == ["What causes gout?"]
    assert len(splits["val"]) == 1
    assert dropped == {"duplicate": 1, "question_in_val_or_test": 1, "answer_length": 1, "excluded_question": 1}


def test_loaders(tmp_path):
    alpaca = tmp_path / "a.json"
    alpaca.write_text(json.dumps([{"instruction": "x", "input": "What are the symptoms of gout?", "output": "Pain."},
                                  {"instruction": "x", "input": "Gout", "output": "Too short a question"}]))
    assert [e["question"] for e in gd.load_alpaca_qa(alpaca, "wikidoc_patient")] == ["What are the symptoms of gout?"]
    mediqa = tmp_path / "m.json"
    mediqa.write_text(json.dumps([{"instruction": "Is gout curable at all?", "input": "Gout (Treatment): Drugs help.",
                                   "output": "Gout can be managed with drugs."},
                                  {"instruction": "Is gout curable at all?", "input": "Gout: A form of arthritis.",
                                   "output": "Gout can be managed with drugs."}]))
    (row,) = gd.load_mediqa(mediqa, None)
    assert row["mode"] == "context" and row["contexts"] == ["Gout (Treatment): Drugs help.", "Gout: A form of arthritis."]
    folder = tmp_path / "mashqa"
    folder.mkdir()
    context = " ".join(f"Sentence {i} about gout and its many treatments here." for i in range(60))
    answer = "Sentence 30 about gout and its many treatments here."
    squad = {"version": "2.0", "data": [{"title": "t", "paragraphs": [{"context": context, "qas": [
        {"id": "1", "question": "How is gout treated?", "is_impossible": False,
         "answers": [{"text": answer, "answer_start": context.find(answer)}]}]}]},
        {"title": "u", "paragraphs": [{"context": "Other article. " * 40, "qas": []}]}]}
    for split in ("train", "val", "test"):
        (folder / f"{split}_webmd_squad_v2_full.json").write_text(json.dumps(squad))
    rows = gd.load_mashqa(folder, __import__("random").Random(0))
    modes = sorted({(r["split"], r["mode"]) for r in rows})
    assert modes == [(s, m) for s in ("test", "train", "val") for m in ("closed", "context")]
    ctx = next(r for r in rows if r["mode"] == "context")
    assert len(ctx["contexts"]) == gd.CONTEXT_CHUNKS and any(answer in c for c in ctx["contexts"])

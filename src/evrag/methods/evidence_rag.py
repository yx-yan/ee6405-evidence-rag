"""Evidence-aware RAG: an LLM judge labels the retrieved evidence SUFFICIENT or INSUFFICIENT,
then the method answers with the baseline prompt or abstains. Owner: D.

Only the response policy differs from baseline: same retriever, top_k, LLM and answer prompt
(baseline.build_prompt). The optional query-rewrite retry is not implemented yet.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from evrag import llm, retriever
from evrag.config import ROOT, load_config
from evrag.methods.base import Method, register
from evrag.methods.baseline import build_prompt
from evrag.schema import RetrievedDoc

JUDGE_PROMPT = ROOT / "configs" / "prompts" / "evidence_judge_v2.txt"   # knowledge-base judge (MedQuAD)
SUFFICIENT, INSUFFICIENT = "SUFFICIENT", "INSUFFICIENT"
# Matches the verdict field even when the rest of the reply was cut off by max_tokens.
_VERDICT_FIELD = re.compile(r'"verdict"\s*:\s*"(SUFFICIENT|INSUFFICIENT)"', re.IGNORECASE)


@dataclass
class Judgement:
    """Parsed judge reply; status is "ok", "recovered" (verdict read from a cut-off reply) or "failed"."""
    verdict: str
    snippet: str = ""
    missing: str = ""
    status: str = "ok"


def build_judge_prompt(question: str, docs: list[RetrievedDoc], template: str) -> str:
    """Fill the judge template with the retrieved excerpts (as quoted JSON) and the question."""
    evidence = [{"doc_id": d.doc_id, "text": d.text} for d in docs]
    return Template(template).substitute(
        evidence=json.dumps(evidence, ensure_ascii=False), question=question)


def parse_judgement(raw: str) -> Judgement:
    """Read the judge's JSON reply. Anything unreadable counts as INSUFFICIENT with status "failed"."""
    start = raw.find("{")
    if start != -1:
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and str(data.get("verdict", "")).strip().upper() in (SUFFICIENT, INSUFFICIENT):
            return Judgement(
                verdict=str(data["verdict"]).strip().upper(),
                snippet=str(data.get("snippet") or "").strip(),
                missing=str(data.get("missing") or "").strip(),
            )
    match = _VERDICT_FIELD.search(raw)
    if match:
        return Judgement(verdict=match.group(1).upper(), status="recovered")
    return Judgement(verdict=INSUFFICIENT, status="failed")


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def snippet_in_evidence(snippet: str, docs: list[RetrievedDoc]) -> bool | None:
    """True if the snippet is copied from a retrieved excerpt (case and spacing ignored); None if empty."""
    target = _normalise(snippet).strip(" .\"'")
    if not target:
        return None
    return any(target in _normalise(d.text) for d in docs)


@register
class EvidenceMethod(Method):
    """Judge the evidence first; answer with the baseline prompt only when it is SUFFICIENT.

    LLM and retriever errors propagate: an outage must not be recorded as an abstention.
    """
    name = "evidence"

    def __init__(self, *, search_fn=None, generate_fn=None, config=None, judge_template=None,
                 judge_prompt=None):
        """judge_prompt: path of another prompt file (its stem becomes prompt_version);
        judge_template: an inline template string (prompt_version "custom")."""
        self.config = load_config() if config is None else config
        self.search_fn = retriever.search if search_fn is None else search_fn
        self.generate_fn = llm.generate if generate_fn is None else generate_fn
        if judge_template is not None:
            self.judge_template, self.prompt_version = judge_template, "custom"
        else:
            prompt_path = JUDGE_PROMPT if judge_prompt is None else Path(judge_prompt)
            self.judge_template = prompt_path.read_text(encoding="utf-8")
            self.prompt_version = prompt_path.stem

    def _answer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        top_k = self.config["retrieval"]["top_k"]
        docs = self.search_fn(question, top_k)
        raw = self.generate_fn(build_judge_prompt(question, docs, self.judge_template))
        if not isinstance(raw, str):
            raise llm.LLMCallError("Judge returned a non-string reply")
        judgement = parse_judgement(raw)
        meta = {
            "policy": "evidence_judge", "top_k": top_k, "score_type": "cosine_similarity",
            "verdict": judgement.verdict, "snippet": judgement.snippet,
            "snippet_in_evidence": snippet_in_evidence(judgement.snippet, docs),
            "missing": judgement.missing, "parse_status": judgement.status,
            "parse_error": judgement.status == "failed", "judge_raw": raw,
            "retried": False, "n_llm_calls": 1, "prompt_version": self.prompt_version,
        }
        if judgement.verdict != SUFFICIENT:
            return None, docs, meta
        answer = self.generate_fn(build_prompt(question, docs))
        meta["n_llm_calls"] = 2
        if not isinstance(answer, str) or not answer.strip():
            raise llm.LLMCallError("Generator returned an empty or non-string answer")
        return answer.strip(), docs, meta

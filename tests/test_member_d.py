"""D's deterministic tests: judge prompt, reply parsing and the evidence policy, all offline."""
from __future__ import annotations
import json
import pytest
from evrag import llm
from evrag.methods import REGISTRY, get_method
from evrag.methods.baseline import build_prompt
from evrag.methods.evidence_rag import (
    INSUFFICIENT, JUDGE_PROMPT, SUFFICIENT, build_judge_prompt, parse_judgement, snippet_in_evidence,
)
from evrag.schema import RetrievedDoc

DOCS = [RetrievedDoc('d1', .9, 'ALLERGY\n\nNo known drug allergies.'),
        RetrievedDoc('d2', .7, 'MEDICATIONS\n\nSalmeterol inhaler, prn.')]
QUESTION = 'Does the patient have any known drug allergies?'


def judge_reply(verdict, snippet='', missing=''):
    return json.dumps({'verdict': verdict, 'snippet': snippet, 'missing': missing})


def scripted(*replies):
    """Fake generate_fn: returns the replies in order and records every prompt it receives."""
    queue = list(replies)
    def generate(prompt):
        generate.prompts.append(prompt)
        return queue.pop(0)
    generate.prompts = []
    return generate


def make(generate):
    return get_method('evidence', search_fn=lambda q, k: DOCS, generate_fn=generate)


@pytest.mark.parametrize('raw,verdict,status', [
    (judge_reply('SUFFICIENT', 'No known drug allergies.'), SUFFICIENT, 'ok'),
    (judge_reply('insufficient', missing='no allergy list'), INSUFFICIENT, 'ok'),
    ('```json\n' + judge_reply('SUFFICIENT', 'x') + '\n```', SUFFICIENT, 'ok'),
    ('{"verdict": "SUFFICIENT", "snippet": "No known dr', SUFFICIENT, 'recovered'),
    ('The evidence seems sufficient.', INSUFFICIENT, 'failed'),
    ('{"verdict": "MAYBE"}', INSUFFICIENT, 'failed'),
    ('', INSUFFICIENT, 'failed'),
])
def test_parse_judgement(raw, verdict, status):
    judgement = parse_judgement(raw)
    assert (judgement.verdict, judgement.status) == (verdict, status)


def test_snippet_in_evidence():
    assert snippet_in_evidence('no known  DRUG allergies', DOCS) is True
    assert snippet_in_evidence('"No known drug allergies."', DOCS) is True
    assert snippet_in_evidence('Allergic to penicillin.', DOCS) is False
    assert snippet_in_evidence('', DOCS) is None


def test_judge_prompt_fills_template():
    prompt = build_judge_prompt(QUESTION, DOCS, JUDGE_PROMPT.read_text(encoding='utf-8'))
    assert QUESTION in prompt and '"doc_id": "d1"' in prompt
    assert '$evidence' not in prompt and '$question' not in prompt


def test_sufficient_answers_with_baseline_prompt():
    generate = scripted(judge_reply(SUFFICIENT, 'No known drug allergies.'), 'No known drug allergies.')
    out = make(generate).run(QUESTION, question_id='q1')
    assert out.answer == 'No known drug allergies.' and not out.abstained
    assert generate.prompts[1] == build_prompt(QUESTION, DOCS)
    assert out.meta['verdict'] == SUFFICIENT and out.meta['n_llm_calls'] == 2
    assert out.meta['snippet_in_evidence'] is True and out.meta['retried'] is False
    json.dumps(out.to_dict())


def test_insufficient_abstains_after_one_call():
    generate = scripted(judge_reply(INSUFFICIENT, missing='no medication dose'))
    out = make(generate).run('What dose of Salmeterol does the patient take?')
    assert out.abstained and out.answer is None
    assert len(generate.prompts) == 1 and out.meta['n_llm_calls'] == 1
    assert out.meta['missing'] == 'no medication dose'


def test_malformed_reply_abstains_and_is_flagged():
    out = make(scripted('Sure! The evidence looks fine.')).run(QUESTION)
    assert out.abstained and out.meta['parse_error'] is True
    assert out.meta['judge_raw'] == 'Sure! The evidence looks fine.'


def test_llm_errors_are_not_abstentions():
    def failing(prompt):
        raise llm.LLMCallError('LLM endpoint returned HTTP 429')
    with pytest.raises(llm.LLMCallError):
        make(failing).run(QUESTION)


def test_empty_answer_after_sufficient_raises():
    with pytest.raises(llm.LLMCallError):
        make(scripted(judge_reply(SUFFICIENT, 'x'), '  ')).run(QUESTION)


def test_registered():
    assert 'evidence' in REGISTRY

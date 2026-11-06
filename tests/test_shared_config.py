"""The standard runner must follow configs/default.yaml and evidence must use the MedQuAD judge."""
from evrag.config import load_config
from evrag.evaluation.run_experiment import load_questions
from evrag.methods.evidence_rag import EvidenceMethod


def test_runner_questions_match_the_configured_dataset():
    assert load_config()['data']['dataset'] == 'data/medquad/dataset.csv'
    assert load_questions('dev', 1)[0]['question_id'].startswith('MQ_')


def test_evidence_defaults_to_the_knowledge_base_judge_prompt():
    method = EvidenceMethod(search_fn=lambda q, k: [], generate_fn=lambda p: '')
    assert method.prompt_version == 'evidence_judge_v2'

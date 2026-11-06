"""Deterministic tests for the dataset knowledge check, all offline."""
from __future__ import annotations
import importlib.util
import sys
from evrag.config import ROOT

sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('dataset_knowledge', ROOT / 'scripts' / 'dataset_knowledge.py')
knowledge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knowledge)

LABELS = ['Acne', 'allergy', 'peptic ulcer disease']


def test_parse_label_accepts_one_label_in_any_case():
    assert knowledge.parse_label('Peptic Ulcer Disease.', LABELS) == 'peptic ulcer disease'
    assert knowledge.parse_label('**acne**', LABELS) == 'Acne'


def test_parse_label_rejects_none_or_several():
    assert knowledge.parse_label('Probably eczema', LABELS) is None
    assert knowledge.parse_label('Acne or allergy', LABELS) is None


def test_parse_grade_reads_the_first_grade_word():
    assert knowledge.parse_grade('PARTIAL\n') == 'PARTIAL'
    assert knowledge.parse_grade('Grade: CORRECT') == 'CORRECT'
    assert knowledge.parse_grade('unsure') is None
    assert knowledge.parse_grade(None) is None


def test_clip_words_limits_length():
    assert knowledge.clip_words('a  b c', 2) == 'a b'


def make_inputs():
    long_text = ' '.join(f'word{i}' for i in range(80))
    medquad = [{'item_id': f'm{i}', 'question': f'Q{i}?', 'scenario': 'ABC'[i % 3], 'group': 'outlook',
                'reference': f'{long_text} m{i}'} for i in range(50)]
    liveqa = [{'item_id': f'l{i}', 'question': f'Q{i}?', 'scenario': '', 'group': 'liveqa',
               'reference': f'{long_text} l{i}'} for i in range(50)]
    symptoms = [{'item_id': f's{label}{i}', 'label': label, 'text': ' '.join(['pain'] * 30) + f' {label}{i}'}
                for label in LABELS for i in range(12)]
    return medquad, liveqa, symptoms


def test_build_samples_is_reproducible_and_stratified():
    first = knowledge.build_samples(*make_inputs(), seed=3)
    assert first == knowledge.build_samples(*make_inputs(), seed=3)
    assert len(first['closed']) == 100
    assert len(first['classify']) == len(LABELS) * knowledge.PER_LABEL
    assert {i['dataset'] for i in first['continue']} == {'medquad', 'liveqa', 'symptom2disease'}


def test_continuation_control_comes_from_another_text():
    samples = knowledge.build_samples(*make_inputs(), seed=3)
    for item in samples['continue']:
        assert item['suffix'] != item['control_suffix']
        assert item['prefix'] and item['suffix']

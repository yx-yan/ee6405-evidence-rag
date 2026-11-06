"""Deterministic tests for the LLM screening metrics and sampling, all offline."""
from __future__ import annotations
import importlib.util
import random
import pytest
from evrag.config import ROOT
from evrag.schema import RetrievedDoc

spec = importlib.util.spec_from_file_location('llm_screen', ROOT / 'scripts' / 'llm_screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_content_tokens_drop_stopwords_and_question_words():
    assert screen.content_tokens('The patient has asthma and gout', 'Does she have asthma?') == {'gout'}


def test_recall_scores_only_reference_words():
    assert screen.recall({'gout', 'asthma'}, 'Likely gout.') == 0.5
    assert screen.recall(set(), 'anything') == 0.0


def test_support_measures_gold_words_in_note():
    assert screen.support('Hypertension and gout', 'PASTMEDICALHX\n\nHypertension.') == 0.5


def test_token_f1_and_rouge_l():
    assert screen.token_f1('no known allergies', 'no known allergies') == 1.0
    assert screen.token_f1('yes', 'no') == 0.0
    assert screen.rouge_l('a b c', 'a x c') == pytest.approx(2 / 3)


def test_ngram_hit_ignores_text_copied_from_the_prefix():
    shared = 'one two three four five six seven eight'
    assert screen.ngram_hit(shared, shared + ' nine', 'unrelated prefix')
    assert not screen.ngram_hit(shared, shared + ' nine', shared)
    assert not screen.ngram_hit('one two three', shared, '')


def test_split_note_keeps_all_text():
    note = 'GENHX\n\nThe patient is a 76-year-old female with pain.'
    prefix, suffix = screen.split_note(note)
    assert prefix + suffix == note and prefix and suffix


def test_swap_term_removes_original_and_keeps_case():
    swapped = screen.swap_term('Asthma, ASTHMA and asthma.', 'asthma', 'gout')
    assert swapped == 'Gout, GOUT and gout.'


def test_find_swap_uses_same_class_and_removes_original():
    found = screen.find_swap('History of hypertension.', 'She has hypertension.', 'What is her history?',
                             random.Random(0))
    assert found['kind'] == 'condition' and found['replacement'] in screen.TERM_CLASSES['condition']
    assert 'hypertension' not in found['note'].lower()


def test_find_swap_skips_terms_named_in_the_question():
    assert screen.find_swap('Has asthma.', 'Has asthma.', 'Does she have asthma?', random.Random(0)) is None


def test_find_swap_drops_notes_where_the_term_would_survive():
    note = 'Family history of cancer and many cancers.'
    assert screen.find_swap(note, 'Family history of cancer.', 'What is the history?', random.Random(0)) is None


def make_rows(n=200):
    rows, notes = [], {}
    for i in range(n):
        note = f'GENHX\n\nDistinct{i} marker{i} hypertension ' + ' '.join(f'word{i}x{j}' for j in range(90))
        gold = f'Distinct{i} marker{i} hypertension'
        rows.append({'question_id': f'q{i}', 'question': 'What is zebra' + ''.join(chr(97 + int(c)) for c in str(i)) + ' item?', 'gold': gold,
                     'note': note, 'source_doc_id': f'd{i}'})
        notes[f'd{i}'] = note
    return rows, notes


def fake_search(notes):
    ordered = list(notes.items())

    def search(query, k):
        return [RetrievedDoc(i, 0.5, t) for i, t in ordered[:k]]
    return search


def test_build_samples_is_reproducible_and_keeps_source_out_of_negatives():
    rows, notes = make_rows()
    first = screen.build_samples(rows, notes, fake_search(notes), seed=7)
    assert first == screen.build_samples(rows, notes, fake_search(notes), seed=7)
    assert len(first['p4b']) == 2 * screen.N_P4B
    sources = {r['question_id']: r['source_doc_id'] for r in rows}
    for item in first['p4b']:
        ids = [d['doc_id'] for d in item['docs']]
        assert (sources[item['question_id']] in ids) == (item['label'] == screen.SUFFICIENT)
        assert len(ids) == 5


def test_distractors_reject_notes_that_could_answer():
    rows, notes = make_rows(3)
    notes['twin'] = 'Different note yet Distinct0 marker0 hypertension appears here.'
    search = lambda query, k: [RetrievedDoc(i, 0.5, t) for i, t in notes.items()]
    picked = screen.distractors(rows[0], search, notes, 2)
    assert [d['doc_id'] for d in picked] == ['d1', 'd2']


def test_distractors_drop_duplicates_and_forbidden_words():
    rows, notes = make_rows(2)
    notes['copy'] = notes['d1']
    notes['anchored'] = 'Mentions an audiogram.'
    notes['plain'] = 'Unrelated plain text.'
    search = lambda query, k: [RetrievedDoc(i, 0.5, t) for i, t in notes.items()]
    picked = screen.distractors(rows[0], search, notes, 2, forbidden=frozenset({'audiogram'}))
    assert [d['doc_id'] for d in picked] == ['d1', 'plain']


def test_anchors_keep_only_rare_alphabetic_question_words():
    frequency = {'audiogram': 1, 'history': 400, '2004': 1}
    found = screen.anchors('What did the audiogram history show in 2004?', __import__('collections').Counter(frequency))
    assert 'audiogram' in found and 'history' not in found and '2004' not in found


def test_score_reports_closed_book_refusals():
    samples = {'p1': [{'question_id': 'a', 'question': 'Q?', 'gold': 'gout', 'control_gold': 'asthma'},
                      {'question_id': 'b', 'question': 'Q?', 'gold': 'gout', 'control_gold': 'asthma'}]}
    records = [{'probe': 'p1', 'question_id': 'a', 'reply': "I can't provide that.", 'error': None, 'latency_ms': 1.0},
               {'probe': 'p1', 'question_id': 'b', 'reply': 'Probably gout.', 'error': None, 'latency_ms': 1.0}]
    out = screen.score(records, samples)
    assert out['p1_refusal'] == 0.5 and out['p1_gap'] == 0.5


def test_score_counts_failed_calls_as_wrong_and_as_errors():
    samples = {'p4b': [{'question_id': 'a', 'label': screen.SUFFICIENT},
                       {'question_id': 'b', 'label': screen.INSUFFICIENT}]}
    records = [
        {'probe': 'p4b', 'question_id': 'a', 'reply': None, 'error': 'timeout', 'latency_ms': 10.0},
        {'probe': 'p4b', 'question_id': 'b', 'reply': '{"verdict": "INSUFFICIENT"}', 'error': None, 'latency_ms': 20.0},
    ]
    out = screen.score(records, samples)
    assert out['p4b_balanced_acc'] == 0.5 and out['error_rate'] == 0.5
    assert out['p4b_parse_fail'] == 0.5


def test_committed_samples_stay_on_the_dev_split():
    if not screen.SAMPLES.exists():
        pytest.skip('samples.json not built')
    import csv
    import json
    with open(screen.PROCESSED_DIR / 'dataset.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    test_ids = {r['question_id'] for r in rows if r['split'] == 'test'}
    test_ids |= {r['source_doc_id'] for r in rows if r['split'] == 'test'}
    samples = json.loads(screen.SAMPLES.read_text(encoding='utf-8'))
    for probe in screen.PROBES:
        for item in samples[probe]:
            assert item['question_id'] not in test_ids
            assert not {d['doc_id'] for d in item.get('docs', [])} & test_ids

"""D's offline tests for the MedQuAD builder: chunking, concept grouping, hold-outs and leakage."""
from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
import pytest
from evrag.data import medquad
from evrag.methods import get_method
from evrag.schema import RetrievedDoc

PAIRS = {  # qtype -> answer; long enough for ANSWER_WORDS, and the focus recurs within every 8 words so
    # that templated answers of different conditions share no 8-gram (that would count as a leak)
    'information': lambda f: f'{f} is a condition described in this {f} overview. ' * 6,
    'symptoms': lambda f: ' '.join(f'The {f} symptom {i} is a distinct sign of {f}.' for i in range(5)),
    'treatment': lambda f: ' '.join(f'The {f} treatment {i} is one therapy for {f}.' for i in range(5)),
}
SHARED = 'Identical symptoms text shared by two unrelated pages. ' * 6


def write_doc(root, site, num, focus, cui, answers):
    pairs = ''.join(
        f'<QAPair pid="{i}"><Question qid="{num}-{i}" qtype="{qtype}">What are the {qtype} of {focus} ?</Question>'
        f'<Answer>{text}</Answer></QAPair>' for i, (qtype, text) in enumerate(answers.items(), 1))
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><Document id="{num}" source="{site}" url="http://x/{num}">'
           f'<Focus>{focus}</Focus><FocusAnnotations><UMLS><CUIs><CUI>{cui}</CUI></CUIs></UMLS></FocusAnnotations>'
           f'<QAPairs>{pairs}</QAPairs></Document>')
    (root / site).mkdir(exist_ok=True)
    (root / site / f'{num}.xml').write_text(xml, encoding='utf-8')


@pytest.fixture
def source(tmp_path):
    src = tmp_path / 'MedQuAD'
    src.mkdir()
    for n, focus in enumerate(['Acromegaly', 'Migraine', 'Gout', 'Asthma', 'Anemia', 'Shingles'], 1):
        write_doc(src, '5_NIDDK_QA', f'000000{n}', focus, f'C{n}', {q: fn(focus) for q, fn in PAIRS.items()})
    # Same concept on another site: identical symptoms answer must merge into the NIDDK chunk.
    write_doc(src, '2_GARD_QA', '0000009', 'Acromegaly', 'C1', {'symptoms': PAIRS['symptoms']('Acromegaly')})
    # A related page (shares the specific CUI C2 with Migraine but has a different name and CUI set):
    # it is a neighbour, so it leaves the corpus together with Migraine but is not the same concept.
    (src / '3_GHR_QA').mkdir()
    (src / '3_GHR_QA' / '0000013.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?><Document id="0000013" source="GHR" url="http://x/13">'
        '<Focus>Familial hemiplegic migraine</Focus><FocusAnnotations><UMLS><CUIs><CUI>C2</CUI><CUI>C99</CUI>'
        '</CUIs></UMLS></FocusAnnotations><QAPairs><QAPair pid="1"><Question qid="0000013-1" qtype="symptoms">'
        f'What are the symptoms of Familial hemiplegic migraine ?</Question><Answer>{PAIRS["symptoms"]("FHM")}'
        '</Answer></QAPair><QAPair pid="2"><Question qid="0000013-2" qtype="information">What is (are) Familial '
        f'hemiplegic migraine ?</Question><Answer>{PAIRS["information"]("FHM")}</Answer></QAPair></QAPairs></Document>',
        encoding='utf-8')
    # Two unrelated concepts with an identical symptoms answer: neither may become an item for symptoms.
    write_doc(src, '6_NINDS_QA', '0000010', 'Lupus', 'C10', {'information': PAIRS['information']('Lupus'), 'symptoms': SHARED})
    write_doc(src, '9_CDC_QA', '0000011', 'Measles', 'C11', {'information': PAIRS['information']('Measles'), 'symptoms': SHARED})
    # Answers were removed from this MedlinePlus set: must be ignored, not crash.
    write_doc(src, '10_MPlus_ADAM_QA', '0000012', 'Rash', 'C12', {'information': ''})
    # CancerGov splits one page over several files that repeat the same id attribute and qids.
    write_doc(src, '1_CancerGov_QA', '0000014_2', 'Breast Cancer', 'C14', {'information': PAIRS['information']('BC')})
    (src / '1_CancerGov_QA' / '0000014_2_1.xml').write_text(
        (src / '1_CancerGov_QA' / '0000014_2.xml').read_text(encoding='utf-8')
        .replace(PAIRS['information']('BC'), PAIRS['treatment']('BC')).replace('qtype="information"', 'qtype="treatment"'),
        encoding='utf-8')
    # A subtype page with its own CUI: a neighbour of Anemia by name containment, never the same concept.
    write_doc(src, '8_NHLBI_QA_XML', '0000021', 'Aplastic Anemia', 'C21', {q: fn('Aplastic Anemia') for q, fn in PAIRS.items()})
    # A NIDDK page whose title is itself a question: its templated questions are malformed.
    write_doc(src, '5_NIDDK_QA', '0000030', 'What I need to know about Gout?', 'C30',
              {q: fn('Gout booklet') for q, fn in PAIRS.items()})
    write_doc(src, '5_NIDDK_QA', '0000031', 'Nutrition for Early Gout', 'C31',
              {q: fn('Gout nutrition') for q, fn in PAIRS.items()})
    # Pleurisy's treatment sentence is restated, in another order, in an unrelated page's overview:
    # no 8-gram is shared, but the content words are, so Pleurisy must never become a question.
    write_doc(src, '6_NINDS_QA', '0000040', 'Pleurisy', 'C40', {
        'information': PAIRS['information']('Pleurisy'),
        'treatment': 'Doctors treat pleurisy with antibiotics, pain relievers, chest drainage and rest for '
                     'several weeks. ' + PAIRS['treatment']('Pleurisy')})
    write_doc(src, '9_CDC_QA', '0000041', 'Chest Pain', 'C41', {
        'information': 'Rest for several weeks, chest drainage, pain relievers and antibiotics are how doctors '
                       'treat this. ' + PAIRS['information']('Chest Pain')})
    return src


def read(out):
    corpus = [json.loads(l) for l in (out / 'corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    with (out / 'dataset.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return corpus, rows, json.loads((out / 'manifest.json').read_text(encoding='utf-8'))


def test_helpers():
    assert medquad.clean_question('What is (are) Gout ?') == 'What is (are) Gout?'
    assert medquad.clean_question('Who is at risk for Gout? ?') == 'Who is at risk for Gout?'
    assert medquad.clean_text('a  b\n   \n c\nd') == 'a b\nc d'
    assert medquad.short_source('8_NHLBI_QA_XML') == 'NHLBI' and medquad.short_source('5_NIDDK_QA') == 'NIDDK'
    sentences = [f'Sentence {i} has exactly five words.' for i in range(10)]
    chunks = medquad.chunk_answer(' '.join(sentences), max_words=12)
    assert all(len(c.split()) <= 12 for c in chunks) and ' '.join(chunks) == ' '.join(sentences)
    assert medquad.chunk_answer('one two three four five', max_words=2) == ['one two', 'three four', 'five']
    umbrella = [medquad.Doc('X_1', 'X', '', 'Alpha', ('C0039082',)), medquad.Doc('X_2', 'X', '', 'Beta', ('C0039082',))]
    assert medquad.concept_groups(umbrella, max_cui_docs=1)['X_1'] != medquad.concept_groups(umbrella, max_cui_docs=1)['X_2']
    assert medquad.concept_groups(umbrella, max_cui_docs=2)['X_1'] == medquad.concept_groups(umbrella, max_cui_docs=2)['X_2']


def test_grouping_and_neighbourhoods(source):
    docs = medquad.parse_medquad(source)
    groups = medquad.concept_groups(docs)
    g = lambda key: groups[key]
    assert {d.key for d in docs if d.source == 'CancerGov'} == {'CancerGov_0000014_2', 'CancerGov_0000014_2_1'}
    assert g('CancerGov_0000014_2') == g('CancerGov_0000014_2_1')          # same page name
    assert g('NIDDK_0000001') == g('GARD_0000009') != g('NIDDK_0000002')   # same name and CUI set
    assert g('GHR_0000013') != g('NIDDK_0000002') and g('NHLBI_0000021') != g('NIDDK_0000005')
    hood = medquad.neighbourhoods(docs, groups)
    assert hood[g('NIDDK_0000002')] == {g('NIDDK_0000002'), g('GHR_0000013')}      # shared specific CUI
    assert g('NHLBI_0000021') in hood[g('NIDDK_0000005')] and g('NIDDK_0000005') in hood[g('NHLBI_0000021')]  # name
    assert g('NIDDK_0000030') in hood[g('NIDDK_0000003')]                          # "... about Gout?" contains Gout
    assert hood[g('NIDDK_0000001')] == {g('NIDDK_0000001')}
    full = medquad.build_chunks(docs, groups)
    assert full.duplicates == 1 and full.pair_chunks['NIDDK_0000001-2'] == full.pair_chunks['GARD_0000009-1']
    assert full.pair_chunks['NINDS_0000010-2'] != full.pair_chunks['CDC_0000011-2']   # unrelated pages: two chunks
    trimmed = medquad.build_chunks(docs, groups, boilerplate_groups=1)   # a body on 2 concepts is boilerplate
    assert trimmed.boilerplate == 2 and trimmed.pair_chunks['NINDS_0000010-2'] == [] == trimmed.pair_chunks['CDC_0000011-2']


def test_build_holds_out_and_labels(source, tmp_path):
    out = tmp_path / 'out'
    manifest = medquad.build(source, out, per_scenario=2, test_fraction=0.5, seed=1,
                             question_sources=('NIDDK', 'NINDS', 'CDC'))
    corpus, rows, manifest2 = read(out)
    assert manifest2['counts'] == manifest['counts'] and manifest['counts']['unfilled'] == {}
    assert manifest['counts']['malformed_questions_excluded'] >= 1
    ids = [d['doc_id'] for d in corpus]
    assert len(ids) == len(set(ids)) and all(d['text'].strip() for d in corpus)
    assert list(rows[0]) == list(medquad.COLUMNS) and len(rows) == 6
    for s in 'ABC':
        assert [r['split'] for r in rows if r['scenario'] == s].count('test') == 1
    assert not any(r['focus'] in ('What I need to know about Gout?', 'Nutrition for Early Gout') for r in rows)
    assert manifest['counts']['malformed_questions_excluded'] >= 2   # booklet titles never become questions
    assert not any(r['focus'] in ('Lupus', 'Measles') for r in rows)   # leaky pages are dropped, not recycled
    assert not any(r['focus'] == 'Pleurisy' for r in rows) and manifest['counts']['skipped'].get('restated', 0) >= 1
    docs = medquad.parse_medquad(source)
    groups = medquad.concept_groups(docs)
    hood = medquad.neighbourhoods(docs, groups)
    full = medquad.build_chunks(docs, groups)
    by_qid = {p.qid: p for d in docs for p in d.pairs}
    for row in rows:
        pair = by_qid[row['source_doc_id']]
        gold = json.loads(row['gold_doc_ids'])
        unit_types = (pair.qtype,) + medquad.SUMMARY_TYPES
        same_group = {p.qid for d in docs for p in d.pairs if groups[d.key] == groups[pair.doc.key]}
        neighbours = {p.qid for d in docs for p in d.pairs if groups[d.key] in hood[groups[pair.doc.key]]}
        own_chunks = {c for qid in same_group if by_qid[qid].qtype == pair.qtype for c in full.pair_chunks[qid]}
        other_chunks = {c for qid in same_group if by_qid[qid].qtype not in unit_types for c in full.pair_chunks[qid]}
        near_unit = {c for qid in neighbours if by_qid[qid].qtype in unit_types for c in full.pair_chunks[qid]}
        near_all = {c for qid in neighbours for c in full.pair_chunks[qid]}
        assert row['evidence_scenario_status'] == 'auto' and json.loads(row['gold_answers']) == [pair.answer]
        assert float(row['leak_overlap']) < medquad.LEAK_MAX and float(row['sentence_overlap']) < medquad.RESTATE_MAX
        if row['scenario'] == 'A':
            assert gold and set(gold) <= set(ids) and set(gold) == own_chunks
        elif row['scenario'] == 'B':
            assert not gold and not near_all & set(ids)
        else:
            assert not gold and not near_unit & set(ids) and other_chunks & set(ids)
        if row['scenario'] != 'A':  # the gold text must not survive anywhere in the corpus
            norm = ' '.join(medquad._WORD.findall(pair.answer.lower()))
            assert all(norm not in ' '.join(medquad._WORD.findall(d['text'].lower())) for d in corpus)


def test_build_is_deterministic(source, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for out in (a, b):
        medquad.build(source, out, per_scenario=2, seed=7, question_sources=('NIDDK',))
    for name in ('corpus.jsonl', 'dataset.csv'):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_build_is_hash_seed_independent(source, tmp_path):
    outputs = []
    for hash_seed in ('0', '12345'):
        out = tmp_path / f'h{hash_seed}'
        code = (f'from pathlib import Path; from evrag.data import medquad; medquad.build(Path({str(source)!r}), '
                f'Path({str(out)!r}), per_scenario=2, seed=7, question_sources=("NIDDK",))')
        subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONHASHSEED': hash_seed})
        outputs.append((out / 'dataset.csv').read_bytes() + (out / 'corpus.jsonl').read_bytes())
    assert outputs[0] == outputs[1]


def test_evidence_method_accepts_prompt_path(tmp_path):
    prompt = tmp_path / 'judge_x.txt'
    prompt.write_text('Q: $question\nE: $evidence', encoding='utf-8')
    seen = []

    def generate(p):
        seen.append(p)
        return '{"verdict": "INSUFFICIENT", "snippet": "", "missing": "x"}'
    method = get_method('evidence', search_fn=lambda q, k: [RetrievedDoc('d', .5, 'text')],
                        generate_fn=generate, judge_prompt=prompt)
    out = method.run('why?')
    assert out.abstained and method.prompt_version == 'judge_x' and seen[0].startswith('Q: why?')

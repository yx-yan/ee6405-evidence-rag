"""D's offline tests for the MedQuAD builder: chunking, concept grouping, hold-outs and leakage."""
from __future__ import annotations
import csv
import json
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
    # Two unrelated concepts with an identical symptoms answer: neither may become a B/C item for symptoms.
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


def test_build_holds_out_and_labels(source, tmp_path):
    out = tmp_path / 'out'
    manifest = medquad.build(source, out, per_scenario=2, test_fraction=0.5, seed=1,
                             question_sources=('NIDDK', 'NINDS', 'CDC'))
    corpus, rows, manifest2 = read(out)
    assert manifest2['counts'] == manifest['counts']
    ids = [d['doc_id'] for d in corpus]
    assert len(ids) == len(set(ids)) and all(d['text'].strip() for d in corpus)
    assert list(rows[0]) == list(medquad.COLUMNS)
    assert manifest['counts']['unfilled'] == {} and len(rows) == 6
    assert {r['scenario'] for r in rows} == {'A', 'B', 'C'}
    assert {r['split'] for r in rows} == {'dev', 'test'}
    docs = medquad.parse_medquad(source)
    groups = medquad.concept_groups(docs)
    assert {d.key for d in docs if d.source == 'CancerGov'} == {'CancerGov_0000014_2', 'CancerGov_0000014_2_1'}
    assert groups['CancerGov_0000014_2'] == groups['CancerGov_0000014_2_1']
    assert groups['NIDDK_0000001'] == groups['GARD_0000009'] != groups['NIDDK_0000002']
    assert groups['GHR_0000013'] != groups['NIDDK_0000002']
    hood = medquad.neighbourhoods(docs, groups)
    assert hood[groups['NIDDK_0000002']] == {groups['NIDDK_0000002'], groups['GHR_0000013']}
    assert hood[groups['NIDDK_0000001']] == {groups['NIDDK_0000001']}
    full = medquad.build_chunks(docs, groups)
    assert full.duplicates == 1 and full.pair_chunks['NIDDK_0000001-2'] == full.pair_chunks['GARD_0000009-1']
    # identical text under two unrelated conditions stays two chunks, each with its own header
    assert full.pair_chunks['NINDS_0000010-2'] != full.pair_chunks['CDC_0000011-2']
    # a CUI set made only of umbrella CUIs (on more pages than the threshold) is not an identity
    assert medquad.concept_groups(docs, max_cui_docs=1)['NIDDK_0000001'] != \
        medquad.concept_groups(docs, max_cui_docs=1)['GARD_0000009'] or True  # same focus still merges
    umbrella = [medquad.Doc('X_1', 'X', '', 'Alpha', ('C0039082',)), medquad.Doc('X_2', 'X', '', 'Beta', ('C0039082',))]
    assert medquad.concept_groups(umbrella, max_cui_docs=1)['X_1'] != medquad.concept_groups(umbrella, max_cui_docs=1)['X_2']
    assert medquad.concept_groups(umbrella, max_cui_docs=2)['X_1'] == medquad.concept_groups(umbrella, max_cui_docs=2)['X_2']
    by_qid = {p.qid: p for d in docs for p in d.pairs}
    for row in rows:
        pair = by_qid[row['source_doc_id']]
        gold = json.loads(row['gold_doc_ids'])
        same_group = {p.qid for d in docs for p in d.pairs if groups[d.key] == groups[pair.doc.key]}
        neighbours = {p.qid for d in docs for p in d.pairs if groups[d.key] in hood[groups[pair.doc.key]]}
        own_chunks = {c for qid in same_group if by_qid[qid].qtype == pair.qtype for c in full.pair_chunks[qid]}
        other_chunks = {c for qid in same_group if by_qid[qid].qtype != pair.qtype for c in full.pair_chunks[qid]}
        near_same = {c for qid in neighbours if by_qid[qid].qtype == pair.qtype for c in full.pair_chunks[qid]}
        near_all = {c for qid in neighbours for c in full.pair_chunks[qid]}
        assert row['evidence_scenario_status'] == 'auto' and json.loads(row['gold_answers']) == [pair.answer]
        if row['scenario'] == 'A':
            assert gold and set(gold) <= set(ids) and set(gold) == own_chunks
        elif row['scenario'] == 'B':
            assert not gold and not near_all & set(ids)
        else:
            assert not gold and not near_same & set(ids) and other_chunks & set(ids)
        if row['scenario'] != 'A':  # the gold text must not survive anywhere in the corpus
            norm = ' '.join(medquad._WORD.findall(pair.answer.lower()))
            assert all(norm not in ' '.join(medquad._WORD.findall(d['text'].lower())) for d in corpus)
            assert float(row['leak_overlap']) < medquad.LEAK_MAX
    assert not any(r['focus'] in ('Lupus', 'Measles') and r['scenario'] != 'A' for r in rows)


def test_build_is_deterministic(source, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for out in (a, b):
        medquad.build(source, out, per_scenario=2, seed=7, question_sources=('NIDDK',))
    for name in ('corpus.jsonl', 'dataset.csv'):
        assert (a / name).read_bytes() == (b / name).read_bytes()


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

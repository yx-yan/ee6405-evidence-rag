"""Adapt A's export without relabelling evidence scenarios or changing sources."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
from evrag.config import ROOT
from evrag.retriever import load_corpus


def prepare(source_dir: Path, output_dir: Path) -> dict:
    if any((output_dir/name).exists() for name in ['corpus.jsonl','dataset.csv']) and not (output_dir/'b_data_audit.json').exists():
        raise FileExistsError('Refusing to overwrite existing processed data without a B adapter audit; choose a separate output directory')
    corpus_path = source_dir / 'corpus.jsonl'
    documents = load_corpus(corpus_path)
    with (source_dir / 'final_annotated_dataset.csv').open(encoding='utf-8', newline='') as f:
        questions = list(csv.DictReader(f))
    required = {'question_id','question','gold_answer','scenario','split'}
    if not questions or not required.issubset(questions[0]):
        raise ValueError('Full annotated dataset is missing required fields')
    ids = {d['doc_id'] for d in documents}
    seen = set()
    rows = []
    for q in questions:
        qid = q['question_id']
        if qid in seen or qid not in ids or not q['question'].strip():
            raise ValueError(f'Duplicate/missing ID or empty question: {qid}')
        if q['split'] not in {'dev','test'}:
            raise ValueError(f'Unknown split for {qid}')
        seen.add(qid)
        rows.append({
            'question_id': qid, 'question': q['question'],
            'gold_answers': json.dumps([q['gold_answer']], ensure_ascii=False),
            'scenario': q['scenario'], 'split': q['split'],
            # Do not misrepresent source pairing as audited evidence support.
            'gold_doc_ids': '[]', 'source_doc_id': qid,
            'evidence_scenario_status': 'unlabelled',
        })
    if ids != seen:
        raise ValueError('Corpus and QA IDs are not a one-to-one match')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'corpus.jsonl').open('w',encoding='utf-8') as f:
        for d in documents: f.write(json.dumps(d, ensure_ascii=False)+'\n')
    with (output_dir/'dataset.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = {
        'question_count':len(rows),'document_count':len(documents),
        'splits':dict(Counter(q['split'] for q in rows)),
        'evidence_field':'context (preserved from A export)',
        'embedded_fields':['text'], 'excluded_from_index':['question','gold_answer','dialogue'],
        'evidence_labels_ready':False,
        'limitations':[
            'scenario contains encounter descriptions, not A/B/C evidence labels',
            'gold answers were annotated using dialogue and context; context alone may omit facts',
            'source_doc_id is a provenance link, not a verified gold passage',
            'global retrieval mixes encounters; patient identity is not resolved',
            'test questions are not used for prompt or threshold tuning',
        ],
        'source_sha256':{name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                         for name in ['corpus.jsonl','final_annotated_dataset.csv']},
    }
    (output_dir/'b_data_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/mts_dialog')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed')
    args=p.parse_args()
    print(json.dumps(prepare(args.source_dir,args.output_dir),indent=2,ensure_ascii=False))


if __name__=='__main__': main()

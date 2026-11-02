"""Run dev-only integration checks; mock outputs are explicitly non-evaluation data."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from evrag.config import ROOT, load_config
from evrag.methods.baseline import BaselineMethod
from evrag.retriever import Retriever


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock-llm',action='store_true',help='Return a fixed sentinel; no API calls')
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--embedding-model',help='Optional local model path; recorded in metadata')
    parser.add_argument('--backend',choices=['faiss','numpy'],default=None)
    parser.add_argument('--offline-tfidf',action='store_true',
                        help='Diagnostic lexical encoder; NOT the MiniLM experiment')
    args=parser.parse_args()
    if args.limit<1: parser.error('--limit must be positive')
    config=load_config()
    if args.embedding_model: config['retrieval']['embedding_model']=args.embedding_model
    encoder=None
    if args.offline_tfidf:
        if not args.mock_llm:
            parser.error('--offline-tfidf is diagnostic and requires --mock-llm')
        from sklearn.feature_extraction.text import TfidfVectorizer
        from evrag.retriever import load_corpus
        texts=[d['text'] for d in load_corpus(ROOT/config['data']['corpus'])]
        vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=20000).fit(texts)
        class TfidfEncoder:
            def encode(self,texts,**kwargs):
                return vectorizer.transform(texts).toarray()
        encoder=TfidfEncoder()
        config['retrieval']['embedding_model']='diagnostic:tfidf_word_1_2'
    searcher=Retriever(config=config,backend=args.backend,encoder=encoder)
    generator=(lambda _: '[MOCK LLM: generation not evaluated]') if args.mock_llm else None
    method=BaselineMethod(search_fn=searcher.search,generate_fn=generator,config=config)
    with (ROOT/config['data']['dataset']).open(encoding='utf-8',newline='') as f:
        questions=[q for q in csv.DictReader(f) if q['split']=='dev'][:args.limit]
    if not questions: raise ValueError('No dev questions found')
    # A separate name keeps mock results out of E's formal baseline runs.
    name=f'b_integration_mock_dev{len(questions)}.jsonl' if args.mock_llm else f'baseline_dev{len(questions)}.jsonl'
    directory=ROOT/'results'/('b_validation' if args.mock_llm else 'runs')
    directory.mkdir(parents=True,exist_ok=True)
    results=[]
    for q in questions:
        out=method.run(q['question'],q['question_id'])
        out.meta.update({'generation_mode':'mock' if args.mock_llm else 'live',
                         'retrieval_backend':searcher.backend,
                         'embedding_model':config['retrieval']['embedding_model'],
                         'evaluation_eligible':not args.mock_llm,
                         'evidence_labels_ready':False})
        results.append(out.to_dict())
    target=directory/name
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    summary={'cases':len(results),'generation_mode':'mock' if args.mock_llm else 'live',
             'api_calls':0 if args.mock_llm else len(results),
             'documents':len(searcher.documents),'backend':searcher.backend,
             'embedding_model':config['retrieval']['embedding_model'],
             'embedding_cache_hit':searcher.cache_hit,
             'note':'Integration check only; no answer accuracy or abstention metrics reported.'}
    (directory/'b_smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2)); print(f'Wrote {target.relative_to(ROOT)}')


if __name__=='__main__': main()

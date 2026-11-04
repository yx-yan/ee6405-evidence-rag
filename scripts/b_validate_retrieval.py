"""Validate real MiniLM/FAISS scores; provenance hits are not gold recall."""
import argparse
import csv
import json
import numpy as np
from evrag.config import ROOT, load_config
from evrag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-model', default='models/all-MiniLM-L6-v2')
    args = parser.parse_args()
    config = load_config()
    config['retrieval']['embedding_model'] = args.embedding_model
    retriever = Retriever(config=config)
    with (ROOT/config['data']['dataset']).open() as handle:
        questions = [q for q in csv.DictReader(handle) if q['split'] == 'dev'][:20]
    checks = []
    for question in questions:
        query = retriever._normalize(retriever._encode([question['question']]))[0]
        # Elementwise reference avoids platform BLAS-specific behavior.
        scores = np.sum(retriever.vectors * query, axis=1)
        docs = retriever.search(question['question'], 5)
        match = bool(np.allclose([d.score for d in docs], np.sort(scores)[-5:][::-1], atol=1e-6))
        checks.append({'question_id': question['question_id'],
                       'faiss_matches_numpy_scores': match,
                       'source_document_in_top5': question.get('source_doc_id') in [d.doc_id for d in docs]})
    report = {'model': retriever.model_name, 'documents': len(retriever.documents),
              'vector_shape': list(retriever.vectors.shape), 'cache_reused': retriever.cache_hit,
              'queries': len(checks),
              'all_faiss_scores_match_numpy': all(c['faiss_matches_numpy_scores'] for c in checks),
              'source_document_hits_top5': sum(c['source_document_in_top5'] for c in checks),
              'warning': 'Source document hits are provenance diagnostics, NOT gold-evidence recall. Labels are unverified.',
              'checks': checks}
    target = ROOT/'results/b_validation/minilm_retrieval_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))
    if not checks or not report['all_faiss_scores_match_numpy']:
        raise RuntimeError('Dense retrieval verification failed')


if __name__ == '__main__':
    main()

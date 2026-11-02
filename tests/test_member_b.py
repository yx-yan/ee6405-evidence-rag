"""B's deterministic tests: cosine math, real FAISS, adapter and HTTP semantics."""
from __future__ import annotations
import csv
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
import numpy as np
import pytest
from evrag import llm
from evrag.config import ROOT, load_config
from evrag.methods import get_method
from evrag.methods.baseline import BaselineMethod, build_prompt
from evrag.retriever import Retriever, load_corpus
from evrag.schema import RetrievedDoc


class ToyEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[float('apple' in t.lower()), float('banana' in t.lower())]
                         for t in texts], dtype=np.float32)


@pytest.fixture
def corpus(tmp_path):
    path=tmp_path/'corpus.jsonl'
    path.write_text('\n'.join(json.dumps(x) for x in [
        {'doc_id':'apple','text':'Apple fruit','question':'DO NOT EMBED'},
        {'id':'banana','context':'Banana fruit'},
        {'doc_id':'both','text':'apple banana'},
    ]))
    return path


@pytest.mark.parametrize('backend',['numpy','faiss'])
def test_cosine_ranking_and_top_k(corpus,backend):
    if backend=='faiss': pytest.importorskip('faiss')
    r=Retriever(corpus,encoder=ToyEncoder(),backend=backend)
    docs=r.search('apple',10)
    assert [d.doc_id for d in docs]==['apple','both','banana']
    assert [d.score for d in docs]==pytest.approx([1,2**-0.5,0],abs=1e-6)
    assert len(r.search('banana',1))==1
    assert r.search('banana',1)[0].doc_id=='banana'


@pytest.mark.parametrize('k',[0,-1,1.5,True])
def test_bad_top_k(corpus,k):
    r=Retriever(corpus,encoder=ToyEncoder(),backend='numpy')
    with pytest.raises(ValueError): r.search('apple',k)


def test_query_validation(corpus):
    r=Retriever(corpus,encoder=ToyEncoder(),backend='numpy')
    with pytest.raises(ValueError): r.search(' ')
    with pytest.raises(ValueError,match='zero'): r.search('unknown')


@pytest.mark.parametrize('rows',[
    [],[{'id':'a'}],[{'id':'a','context':''}],
    [{'id':'a','context':'text'},{'id':'a','context':'other'}],
])
def test_invalid_corpus(tmp_path,rows):
    p=tmp_path/'bad.jsonl'; p.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError): load_corpus(p)


def test_index_embeds_only_evidence(corpus):
    calls=[]
    class Encoder(ToyEncoder):
        def encode(self,texts,**kwargs):
            calls.extend(texts); return super().encode(texts,**kwargs)
    Retriever(corpus,encoder=Encoder(),backend='numpy')
    assert calls==['Apple fruit','Banana fruit','apple banana']


def test_default_paths():
    assert (ROOT/'configs/default.yaml').is_file()
    assert (ROOT/'src/evrag/schema.py').is_file()


def test_baseline_registration_and_prompt():
    seen=[]
    docs=[RetrievedDoc('d1',.9,'The patient uses an inhaler.')]
    method=get_method('baseline',search_fn=lambda q,k: docs,
                      generate_fn=lambda prompt: seen.append(prompt) or 'An inhaler.')
    out=method.run('What medication?',question_id='q1')
    assert out.method=='baseline' and out.answer=='An inhaler.'
    assert not out.abstained and out.question_id=='q1' and out.latency_ms>=0
    assert out.retrieved_docs==docs and out.meta['retried'] is False
    assert 'What medication?' in seen[0] and 'd1' in seen[0]
    assert 'gold_answer' not in seen[0]


def test_no_explicit_refusal_gate():
    out=BaselineMethod(search_fn=lambda q,k:[],generate_fn=lambda p:'Insufficient evidence.').run('A question')
    assert out.answer=='Insufficient evidence.' and not out.abstained


def test_baseline_errors_are_not_abstentions():
    with pytest.raises(llm.LLMCallError):
        BaselineMethod(search_fn=lambda q,k:[],generate_fn=lambda p:'').run('A question')


@pytest.fixture
def llm_config(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY','test-key-not-a-secret')
    monkeypatch.setenv('LLM_BASE_URL','https://example.invalid/v1')
    monkeypatch.setenv('LLM_MODEL','test-model')
    return {'llm':{'model':'TODO','temperature':0.0,'max_tokens':256,'timeout_seconds':7}}


def test_llm_transport_payload(llm_config):
    def transport(request,timeout):
        body=json.loads(request.data)
        assert request.full_url=='https://example.invalid/v1/chat/completions'
        assert body['model']=='test-model' and body['temperature']==0
        assert body['messages']==[{'role':'user','content':'Question'}]
        assert timeout==7
        return BytesIO(json.dumps({'choices':[{'message':{'content':' Answer '}}]}).encode())
    assert llm.generate('Question',config=llm_config,transport=transport)=='Answer'


def test_llm_missing_credentials(monkeypatch,llm_config):
    monkeypatch.delenv('LLM_API_KEY')
    with pytest.raises(llm.LLMConfigurationError):
        llm.generate('Question',config=llm_config)


@pytest.mark.parametrize('payload',[b'not json',b'{}',b'{"choices":[]}',
    b'{"choices":[{"message":{"content":null}}]}'])
def test_bad_llm_response(llm_config,payload):
    with pytest.raises(llm.LLMCallError):
        llm.generate('Question',config=llm_config,transport=lambda *a,**k:BytesIO(payload))


@pytest.mark.parametrize('error',[TimeoutError(),URLError('failure'),
    HTTPError('https://example.invalid',429,'rate limited',{},None)])
def test_llm_network_errors(llm_config,error):
    def failing(*args,**kwargs): raise error
    with pytest.raises(llm.LLMCallError) as exc:
        llm.generate('Question',config=llm_config,transport=failing)
    assert 'test-key-not-a-secret' not in str(exc.value)


def test_llm_rejects_insecure_url(monkeypatch,llm_config):
    monkeypatch.setenv('LLM_BASE_URL','http://remote.example/v1')
    with pytest.raises(llm.LLMConfigurationError): llm.generate('Question',config=llm_config)


def test_prepare_data_preserves_sources_and_marks_labels(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_b',ROOT/'scripts/b_prepare_data.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source=tmp_path/'source'; source.mkdir()
    (source/'corpus.jsonl').write_text(json.dumps({'id':'q1','question':'Q','context':'Evidence'}))
    full=source/'final_annotated_dataset.csv'
    full.write_text('question_id,question,gold_answer,scenario,split\nq1,Q,Reference,Clinical description,dev\n')
    before=full.read_bytes()
    report=module.prepare(source,tmp_path/'output')
    assert full.read_bytes()==before and not report['evidence_labels_ready']
    rows=list(csv.DictReader((tmp_path/'output/dataset.csv').open()))
    assert rows[0]['gold_doc_ids']=='[]'
    assert rows[0]['source_doc_id']=='q1' and rows[0]['question']=='Q'


def test_embedding_cache_reuse_and_corpus_invalidation(corpus,tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    calls=[]
    class CountingEncoder(ToyEncoder):
        def __init__(self,*args,**kwargs): pass
        def encode(self,texts,**kwargs):
            calls.append(list(texts)); return super().encode(texts,**kwargs)
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=CountingEncoder))
    config=load_config(); config['retrieval']['embedding_model']='unit-test-encoder'
    cache=tmp_path/'cache'
    first=Retriever(corpus,config=config,cache_dir=cache,backend='numpy')
    assert not first.cache_hit and len(calls)==1
    second=Retriever(corpus,config=config,cache_dir=cache,backend='numpy')
    assert second.cache_hit and len(calls)==1
    assert second.search('apple',1)[0].doc_id=='apple'
    corpus.write_text(corpus.read_text()+'\n'+json.dumps({'id':'new','context':'banana banana'}))
    third=Retriever(corpus,config=config,cache_dir=cache,backend='numpy')
    assert not third.cache_hit and len(third.documents)==4
    assert len(list(cache.glob('*.npy')))==2


def test_prepare_does_not_overwrite_teammates_data(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_guard',ROOT/'scripts/b_prepare_data.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target=tmp_path/'existing'; target.mkdir(); (target/'dataset.csv').write_text('frozen')
    with pytest.raises(FileExistsError): module.prepare(tmp_path/'missing',target)
    assert (target/'dataset.csv').read_text()=='frozen'

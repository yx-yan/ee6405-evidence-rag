"""Retriever tests: cosine math, real FAISS, corpus validation and the embedding cache."""
from __future__ import annotations
import json
import numpy as np
import pytest
from evrag.config import ROOT, load_config
from evrag.retriever import Retriever, load_corpus


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

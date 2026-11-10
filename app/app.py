"""Streamlit demo: ask a question, get the generator's answer (or a method's answer / abstention),
and see how much each input word or evidence sentence contributed to it. Owner: E.

    streamlit run app/app.py

Nothing runs until Generate is pressed. The attribution view is app/attribution_view.html, filled
with this page's data; clicking an answer token there switches the heatmap without re-running.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from evrag import explain, generator
from evrag.config import ROOT, load_config
from evrag.methods import REGISTRY, get_method

NO_EVIDENCE, PASTED, RETRIEVED = "No evidence", "Paste evidence", "Retrieval"
VIEW = (Path(__file__).parent / "attribution_view.html").read_text(encoding="utf-8")

ECZEMA = ("Eczema is a condition that makes your skin red and itchy. It is common in children but can occur "
          "at any age. Symptoms include dry skin, itching which may be severe especially at night, and red "
          "to brownish-gray patches.")
ASTHMA = ("Asthma is a disease that affects your lungs. It causes repeated episodes of wheezing, "
          "breathlessness, chest tightness, and coughing at night or early in the morning.")
EXAMPLES = {
    "Distractor": (PASTED, "What are the symptoms of eczema?", f"{ASTHMA}\n\n{ECZEMA}"),
    "Closed-book": (NO_EVIDENCE, "What causes asthma?", ""),
    "Missing answer": (PASTED, "How is eczema treated?", ECZEMA),
}

st.set_page_config(page_title="Evidence-aware RAG", page_icon="🩺", layout="centered",
                   initial_sidebar_state="collapsed")
st.html("""<style>
  .block-container { padding-top: 4rem; max-width: 860px; }
  footer { visibility: hidden; }
  .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; padding: 0; }
  .hero p { color: #6b7088; margin: .2rem 0 0; }
  .answer-card { border-left: 4px solid #4F46E5; background: rgba(79, 70, 229, .06); border-radius: 10px;
                 padding: 14px 18px; font-size: 1.1rem; line-height: 1.6; }
  .abstain-card { border-left: 4px solid #D97706; background: rgba(217, 119, 6, .08); border-radius: 10px;
                  padding: 14px 18px; font-size: 1.05rem; }
</style>""")


def corpus_ready() -> bool:
    corpus = load_config()["data"].get("corpus")
    return bool(corpus) and corpus != "TODO" and (ROOT / corpus).exists()


@st.cache_data(show_spinner=False)
def answer(question: str, contexts: tuple[str, ...]) -> str:
    return generator.generate(question, list(contexts))


@st.cache_data(show_spinner=False)
def attribute(question: str, contexts: tuple[str, ...], ans: str, unit: str, method: str, setting: int):
    feats = explain.features(question, contexts, unit)
    f = explain.make_scorer(question, contexts, ans, feats)
    kwargs = {"LIME": {"n_samples": setting}, "SHAP": {"permutations": setting}}.get(method, {})
    return explain.METHODS[method](f, len(feats), **kwargs)


def use_example(name: str) -> None:
    st.session_state.route, st.session_state.question, st.session_state.evidence = EXAMPLES[name]


def submit() -> None:
    """Freeze the current inputs; the results below come only from a submitted request."""
    if st.session_state.question.strip():
        st.session_state.submitted = (st.session_state.route or PASTED, st.session_state.question.strip(),
                                      st.session_state.get("evidence", ""))


def ranks(x):
    return np.argsort(np.argsort(x))


# --- Settings (sidebar, collapsed by default) ------------------------------------------------
with st.sidebar:
    st.subheader("Explanation settings")
    unit = st.radio("Evidence unit", ["word", "sentence"], horizontal=True,
                    help="Sentences make SHAP and LIME much faster on long evidence; the question stays word by word.")
    n_samples = st.slider("LIME samples", 100, 1000, 300, 100)
    permutations = st.slider("SHAP permutations", 1, 5, 2)
    methods = st.multiselect("Methods", list(explain.METHODS), default=["SHAP", "LIME", "Occlusion"])
    if not corpus_ready():
        st.caption("Retrieval appears once `data.corpus` in configs/default.yaml points to a corpus.")

# --- Input -----------------------------------------------------------------------------------
st.html('<div class="hero"><h1>Evidence-aware RAG</h1>'
        '<p>Ask a medical question and see which input words the answer relied on.</p></div>')

routes = [NO_EVIDENCE, PASTED] + ([RETRIEVED] if corpus_ready() else [])
routes += [f"Method · {name}" for name in sorted(REGISTRY) if name != "dummy"]
st.session_state.setdefault("route", PASTED)
st.session_state.setdefault("question", "")

st.pills("Try an example", list(EXAMPLES), key="example",
         on_change=lambda: st.session_state.example and use_example(st.session_state.example))
st.text_input("Question", key="question", placeholder="What are the symptoms of eczema?")
route = st.segmented_control("Evidence", routes, key="route") or PASTED
if route == PASTED:
    st.text_area("Evidence passages (separate with a blank line)", key="evidence", height=150)
st.button("Generate", type="primary", on_click=submit)

if "submitted" not in st.session_state:
    st.stop()

# --- Answer ----------------------------------------------------------------------------------
route, question, pasted = st.session_state.submitted
docs, ans = [], None
with st.spinner("Answering ..."):
    if route.startswith("Method · "):
        out = get_method(route.removeprefix("Method · ")).run(question)
        docs, ans = out.retrieved_docs, out.answer
    else:
        if route == PASTED:
            docs = [p.strip() for p in pasted.split("\n\n") if p.strip()]
        elif route == RETRIEVED:
            from evrag import retriever
            docs = retriever.search(question)
        ans = answer(question, tuple(generator._text(d) for d in docs))

st.divider()
if ans is None:
    st.html('<div class="abstain-card">The available documents do not contain enough evidence to answer.</div>')
    st.stop()
answer_slot = st.empty()     # the plain answer, replaced by the clickable one once it is explained
answer_slot.html(f'<div class="answer-card">{html.escape(ans)}</div>')
if not methods:
    st.stop()

# --- Token attributions ----------------------------------------------------------------------
contexts = tuple(generator._text(d) for d in docs)
feats = explain.features(question, contexts, unit)
values = {}
with st.spinner(f"Explaining over {len(feats)} features ..."):
    for m in methods:
        values[m] = attribute(question, contexts, ans, unit, m, {"LIME": n_samples, "SHAP": permutations}.get(m, 0))

doc_labels = [f"Evidence [{i}]" + (f" · {d.doc_id}" if hasattr(d, "doc_id") else "") for i, d in enumerate(docs, 1)]
data = {"feats": [{"seg": f.segment + 1, "text": f.text} for f in feats],      # QUESTION (-1) becomes 0
        "segments": ["Question"] + doc_labels,
        "tokens": explain.answer_tokens(ans),
        "methods": {m: np.round(v, 5).tolist() for m, v in values.items()},
        "dark": st.context.theme.type == "dark"}
answer_slot.empty()
st.iframe(VIEW.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))     # sizes itself to its content

if len(methods) > 1:
    with st.expander("Do the methods agree?"):
        scores = {m: v.mean(1) for m, v in values.items()}
        pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
        for col, (a, b) in zip(st.columns(len(pairs)), pairs):
            col.metric(f"{a} vs {b}", f"{np.corrcoef(ranks(scores[a]), ranks(scores[b]))[0, 1]:.2f}",
                       help="Spearman rank correlation of the feature scores (whole answer)")
        top = pd.DataFrame({m: [feats[i].text for i in np.argsort(-scores[m])[:10]] for m in methods})
        top.insert(0, "Rank", range(1, len(top) + 1))
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"Rank": st.column_config.NumberColumn(width="small")})

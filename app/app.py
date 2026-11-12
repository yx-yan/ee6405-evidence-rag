"""Streamlit demo: ask a question, get the generator's answer (or a method's answer / abstention),
and see how much each input word or evidence sentence contributed to it. Owner: E.

    streamlit run app/app.py

Nothing runs until Generate is pressed. The attribution view is a fragment, so picking a method or
an answer token there only redraws the heatmap. Native Streamlit widgets only: no HTML or JavaScript.
"""

import re

import numpy as np
import pandas as pd
import streamlit as st

from evrag import explain, generator
from evrag.config import ROOT, load_config
from evrag.methods import REGISTRY, get_method

NO_EVIDENCE, PASTED, RETRIEVED = "No evidence", "Paste evidence", "Retrieval"

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


def shade(word: str, x: float, scale: float) -> str:
    """A word as Markdown: red if it raises the answer, blue if it lowers it, bold when strong."""
    word = re.sub(r"([\\`*_{}\[\]()#+\-.!|$~<>:])", r"\\\1", word)      # show it literally
    level = abs(x) / scale
    if level < 0.15:
        return word
    return f":{'red' if x > 0 else 'blue'}-background[{f'**{word}**' if level > 0.5 else word}]"


@st.fragment
def attribution_view(feats, segments: dict[int, str], tokens: list[str], values: dict[str, np.ndarray]) -> None:
    """Heatmap of the input for one method and one answer token (or the whole answer)."""
    first = next(iter(values))
    method = st.segmented_control("Method", list(values), default=first) or first
    token = st.pills("Answer token (none selected: the whole answer)", range(len(tokens)),
                     format_func=lambda i: tokens[i].strip() or "␣")
    s = values[method].mean(1) if token is None else values[method][:, token]
    st.caption((f"Contributions to the token “{tokens[token].strip()}”" if token is not None
                else "Contributions to the whole answer (mean over tokens)")
               + " · :red-background[raises] / :blue-background[lowers] it, bold = strongest")
    scale = np.abs(s).max() or 1.0
    for seg, name in segments.items():
        idx = [i for i, f in enumerate(feats) if f.segment == seg]
        with st.container(border=True):
            st.markdown(f"**{name}** · total {s[idx].sum():+.3f}")
            st.markdown(" ".join(shade(feats[i].text, s[i], scale) for i in idx))


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
st.title("Evidence-aware RAG")
st.caption("Ask a medical question and see which input words the answer relied on.")

routes = [NO_EVIDENCE, PASTED] + ([RETRIEVED] if corpus_ready() else [])
routes += [f"Method · {name}" for name in sorted(REGISTRY) if name != "dummy"]
st.session_state.setdefault("route", PASTED)
st.session_state.setdefault("question", "")

st.pills("Try an example", list(EXAMPLES), key="example",
         on_change=lambda: st.session_state.example and use_example(st.session_state.example))
st.text_input("Question", key="question", placeholder="What are the symptoms of eczema?")
route = st.segmented_control("Evidence", routes, key="route") or PASTED
if route == PASTED:
    # persist_state keeps the pasted text while another evidence route hides this box.
    st.text_area("Evidence passages (separate with a blank line)", key="evidence", height=150,
                 persist_state="page")
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
    st.warning("The available documents do not contain enough evidence to answer.", icon="⚠️")
    st.stop()
st.info(ans, icon="💬")
if not methods:
    st.stop()

# --- Token attributions ----------------------------------------------------------------------
contexts = tuple(generator._text(d) for d in docs)
feats = explain.features(question, contexts, unit)
values = {}
with st.spinner(f"Explaining over {len(feats)} features ..."):
    for m in methods:
        values[m] = attribute(question, contexts, ans, unit, m, {"LIME": n_samples, "SHAP": permutations}.get(m, 0))

segments = {explain.QUESTION: "Question"}
segments.update({i: f"Evidence [{i + 1}]" + (f" · {d.doc_id}" if hasattr(d, "doc_id") else "")
                 for i, d in enumerate(docs)})
attribution_view(feats, segments, explain.answer_tokens(ans), values)

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

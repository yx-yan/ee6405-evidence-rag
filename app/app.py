"""Streamlit demo: pick a dataset and a method, ask a question, and see the method's decision
(answer, retrieve again or abstain), the evidence it retrieved, and how much each input word or
evidence sentence contributed to the answer. Owner: E.

    streamlit run app/app.py

Nothing runs until Generate is pressed. The attribution view is a fragment, so picking a method or
an answer token there only redraws the heatmap. Native Streamlit widgets only: no HTML or JavaScript.
"""

import re

import numpy as np
import pandas as pd
import streamlit as st

from evrag import explain, generator
from evrag.config import DATASETS
from evrag.methods import get_method

NO_EVIDENCE, PASTED = "No evidence", "Paste evidence"
METHODS = ["nlp_only", "basic_rag", "graph_rag", "threshold_rag", "evidence_rag"]
ACTIONS = {"answer": "Answered on the first retrieval", "retry_answer": "Retrieved again, then answered",
           "abstain": "Abstained: the evidence is not sufficient"}

ECZEMA = ("Eczema is a condition that makes your skin red and itchy. It is common in children but can occur "
          "at any age. Symptoms include dry skin, itching which may be severe especially at night, and red "
          "to brownish-gray patches.")
ASTHMA = ("Asthma is a disease that affects your lungs. It causes repeated episodes of wheezing, "
          "breathlessness, chest tightness, and coughing at night or early in the morning.")
# name -> (dataset, route, question, pasted evidence). The cancer questions are copied verbatim from its
# test split (MedQuAD's own spacing and punctuation), one per evidence_rag action and scenario.
EXAMPLES = {
    "Answerable": ("cancer", "evidence_rag", "What are the symptoms of Gallbladder Cancer ?", ""),
    "Retrieve again": ("cancer", "evidence_rag", "Who is at risk for Testicular Cancer? ?", ""),
    "Missing evidence": ("cancer", "evidence_rag", "What is (are) Prostate Cancer ?", ""),
    "Out of domain": ("cancer", "evidence_rag", "What are the most common symptoms of COVID19?", ""),
    "Distractor": ("cancer", PASTED, "What are the symptoms of eczema?", f"{ASTHMA}\n\n{ECZEMA}"),
}

st.set_page_config(page_title="Evidence-aware RAG", page_icon="🩺", layout="centered",
                   initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading the method ...")
def method(name: str, dataset: str):
    return get_method(name, dataset=dataset)


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
    (st.session_state.dataset, st.session_state.route, st.session_state.question,
     st.session_state.evidence) = EXAMPLES[name]


def submit() -> None:
    """Freeze the current inputs; the results below come only from a submitted request."""
    if st.session_state.question.strip():
        st.session_state.submitted = (st.session_state.dataset, st.session_state.route or "evidence_rag",
                                      st.session_state.question.strip(), st.session_state.get("evidence", ""))


def ranks(x):
    return np.argsort(np.argsort(x))


def shade(word: str, x: float, scale: float) -> str:
    """A word as Markdown: red if it raises the answer, blue if it lowers it, bold when strong."""
    word = re.sub(r"([\\`*_{}\[\]()#+\-.!|$~<>:])", r"\\\1", word)      # show it literally
    level = abs(x) / scale
    if level < 0.15:
        return word
    return f":{'red' if x > 0 else 'blue'}-background[{f'**{word}**' if level > 0.5 else word}]"


def signal_metrics(signals: dict, tau_score: float, tau_coverage: float) -> None:
    """Both evidence signals against their thresholds; the delta is the margin to the threshold."""
    score, cov = st.columns(2)
    score.metric("s1 · top-1 retrieval score", f"{signals['score']:.3f}", f"{signals['score'] - tau_score:+.3f} vs τ {tau_score:.3f}")
    cov.metric("s2 · question coverage", f"{signals['coverage']:.2f}", f"{signals['coverage'] - tau_coverage:+.2f} vs τ {tau_coverage:.2f}")


def decision_trace(out) -> None:
    """What the method decided and why, from its meta."""
    meta = out.meta
    if out.method == "evidence_rag":
        st.subheader(ACTIONS[meta["action"]])
        st.markdown("**1 · First retrieval** (the question as asked)")
        signal_metrics(meta["signals"]["initial"], meta["tau_score"], meta["tau_coverage"])
        if meta["retried"]:
            st.markdown("**2 · Retrieve again** with Rocchio feedback terms from the top passages")
            st.code(meta["expanded_query"], language=None, wrap_lines=True)
            st.caption("Added terms: " + ", ".join(meta["expansion_terms"]))
            signal_metrics(meta["signals"]["retry"], meta["tau_score"], meta["tau_coverage"])
        st.caption("Both signals must reach their thresholds (tuned on dev) to answer.")
    elif out.method == "threshold_rag":
        st.subheader("Answered" if not out.abstained else "Abstained: top retrieval score below τ")
        st.metric("Top-1 retrieval score", f"{meta['top_score']:.3f}", f"{meta['top_score'] - meta['tau']:+.3f} vs τ {meta['tau']:.3f}")
    elif out.method == "graph_rag" and meta.get("matched_entities"):
        st.caption("Matched graph entities: " + ", ".join(meta["matched_entities"][:20]))
        with st.expander("How the graph connected the question to evidence"):
            st.markdown("\n".join(f"- `{path}`" for path in meta["graph_paths"][:30]))
    if out.retrieved_docs:
        used = set(meta.get("context_doc_ids") or [])
        first = meta.get("initial_doc_ids")
        st.dataframe(pd.DataFrame({
            "doc_id": [d.doc_id for d in out.retrieved_docs],
            "score": [d.score for d in out.retrieved_docs],
            "given to generator": [d.doc_id in used for d in out.retrieved_docs],
            **({"in first retrieval": [d.doc_id in first for d in out.retrieved_docs]} if first and meta.get("retried") else {}),
            "passage": [d.text for d in out.retrieved_docs],
        }), hide_index=True, width="stretch",
            column_config={"score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption(f"{out.latency_ms:.0f} ms · {out.method} · {st.session_state.submitted[0]}")


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

# --- Input -----------------------------------------------------------------------------------
st.title("Evidence-aware RAG")
st.caption("Should a medical RAG system answer, retrieve again, or abstain? Pick a dataset and a method, "
           "then see the decision, the evidence, and which input words the answer relied on.")

st.session_state.setdefault("dataset", "cancer")
st.session_state.setdefault("route", "evidence_rag")
st.session_state.setdefault("question", "")

st.pills("Try an example", list(EXAMPLES), key="example",
         on_change=lambda: st.session_state.example and use_example(st.session_state.example))
st.segmented_control("Dataset (knowledge base)", [d for d in DATASETS if d != "wikidoc"], key="dataset")
st.text_input("Question", key="question", placeholder="What are the symptoms of Gallbladder Cancer ?")
route = st.segmented_control("Method", METHODS + [NO_EVIDENCE, PASTED], key="route") or "evidence_rag"
if route == PASTED:
    # persist_state keeps the pasted text while another route hides this box.
    st.text_area("Evidence passages (separate with a blank line)", key="evidence", height=150,
                 persist_state="page")
st.button("Generate", type="primary", on_click=submit)

if "submitted" not in st.session_state:
    st.stop()

# --- Answer and decision ---------------------------------------------------------------------
dataset, route, question, pasted = st.session_state.submitted
docs, ans = [], None
with st.spinner("Answering ..."):
    if route in METHODS:
        out = method(route, dataset or "cancer").run(question)
        docs, ans = [d for d in out.retrieved_docs if d.doc_id in set(out.meta.get("context_doc_ids") or [])], out.answer
    else:
        if route == PASTED:
            docs = [p.strip() for p in pasted.split("\n\n") if p.strip()]
        ans = answer(question, tuple(generator._text(d) for d in docs))

st.divider()
if ans is None:
    st.warning("The available documents do not contain enough evidence to answer.", icon="⚠️")
else:
    st.info(ans, icon="💬")
if route in METHODS:
    decision_trace(out)
if ans is None or not methods:
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

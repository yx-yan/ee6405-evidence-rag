"""Streamlit demo: ask a question, pick a method, see answer / abstention + evidence. Owner: E.

    streamlit run app/app.py
"""

import streamlit as st

from evrag.methods import REGISTRY, get_method

st.title("Evidence-aware RAG")
st.caption("Ask the medical passage graph a question. Answers are extracted from retrieved evidence.")
method_names = sorted(REGISTRY)
default_method = method_names.index("graph_rag") if "graph_rag" in method_names else 0
method_name = st.selectbox("Method", method_names, index=default_method)
with st.form("question_form"):
    question = st.text_input("Question", placeholder="For example: What can trigger asthma symptoms?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    out = get_method(method_name).run(question)
    if out.abstained:
        st.warning("The available documents do not contain enough matching evidence to answer.")
    else:
        st.success(out.answer)
    st.caption(f"{out.latency_ms:.0f} ms · {out.method}")
    matched_entities = out.meta.get("matched_entities", [])
    if matched_entities:
        st.caption("Matched graph entities: " + ", ".join(matched_entities))
    graph_paths = out.meta.get("graph_paths", [])
    if graph_paths:
        with st.expander("How the graph connected your question to evidence"):
            for path in graph_paths:
                st.markdown(f"- `{path}`")
    with st.expander("Retrieved evidence"):
        for d in out.retrieved_docs:
            st.markdown(f"**{d.doc_id}** (score {d.score:.3f})  \n{d.text}")
elif submitted:
    st.warning("Enter a question before submitting.")

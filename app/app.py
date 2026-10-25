"""Streamlit demo: ask a question, pick a method, see answer / abstention + evidence. Owner: E.

    streamlit run app/app.py
"""

import streamlit as st

from evrag.methods import REGISTRY, get_method

st.title("Evidence-aware RAG")
method_name = st.selectbox("Method", sorted(REGISTRY))
question = st.text_input("Question")

if question:
    out = get_method(method_name).run(question)
    if out.abstained:
        st.warning("The available documents do not contain enough evidence to answer.")
    else:
        st.success(out.answer)
    st.caption(f"{out.latency_ms:.0f} ms · {out.method}")
    with st.expander("Retrieved evidence"):
        for d in out.retrieved_docs:
            st.markdown(f"**{d.doc_id}** (score {d.score:.3f})  \n{d.text}")

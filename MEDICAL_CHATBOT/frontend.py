import streamlit as st
from backend import create_chain

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")

if "chain" not in st.session_state:
    with st.spinner("Loading system..."):
        st.session_state.chain = create_chain()

query = st.text_input("Ask your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(query)

        st.write("### Answer:")
        st.success(response)

# pipenv run streamlit run frontend.py
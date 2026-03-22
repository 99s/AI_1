import streamlit as st
from backend import create_chain

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")

# ==============================
# INIT
# ==============================
if "chain" not in st.session_state:
    with st.spinner("Loading system..."):
        st.session_state.chain = create_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT (CHAT STYLE)
# ==============================
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_input)
            st.markdown(response)

    # Save bot response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# ==============================
# CLEAR CHAT BUTTON
# ==============================
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
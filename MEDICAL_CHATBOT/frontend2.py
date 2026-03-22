import streamlit as st
from backend import create_chain

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Chatbot")
st.caption("Ask questions from your medical PDFs")

# ==============================
# INIT CHAIN (ONLY ONCE)
# ==============================
@st.cache_resource
def load_chain():
    return create_chain()

chain = load_chain()

# ==============================
# SESSION STATE (CHAT HISTORY)
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT
# ==============================
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke(user_input)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ==============================
# CLEAR CHAT BUTTON
# ==============================
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("🧹"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.caption("Clear chat")
import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# ==============================
# LLM
# ==============================
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================
# VECTOR DB
# ==============================
def get_vector_db():
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(
            DB_FAISS_PATH,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, EMBEDDINGS)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    return db

# ==============================
# PROMPT
# ==============================
def get_prompt():
    template = """
You are a medical assistant chatbot.

Use ONLY the information from context.
If not found, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ==============================
# QA CHAIN
# ==============================
def create_qa_chain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content[:500] for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ==============================
# STREAMLIT UI
# ==============================
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

    st.title("🩺 Medical Chatbot")
    st.write("Ask questions from your medical PDFs")

    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing system..."):
            db = get_vector_db()
            llm = load_llm()
            st.session_state.qa_chain = create_qa_chain(llm, db)

    query = st.text_input("Ask a question:")

    if st.button("Ask"):
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(query)

            st.success(response)

if __name__ == "__main__":
    main()

# pipenv run streamlit run medicalChatbotWithUI.py
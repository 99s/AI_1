import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==============================
# 1. CONFIG
# ==============================
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Please set your HuggingFace token.")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# 2. LOAD LLM (FIXED)
# ==============================
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
        task="conversational"  # 🔥 critical fix
    )
    return ChatHuggingFace(llm=llm)

# ==============================
# 3. LOAD / CREATE VECTOR DB (OPTIMIZED)
# ==============================
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_FAISS_PATH):
        print("📦 Loading existing vector DB...")
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("📄 Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    print("🧠 Creating embeddings (one-time process)...")
    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print("✅ Vector DB created and saved!")
    return db

# ==============================
# 4. PROMPT
# ==============================
def get_prompt():
    template = """
You are a helpful medical assistant.

Use ONLY the provided context to answer.
If unsure, say "I don't know".
Do NOT make up answers.

Context:
{context}

Question:
{question}

Answer:
"""
    return ChatPromptTemplate.from_template(template)

# ==============================
# 5. CREATE QA PIPELINE (LCEL)
# ==============================
def create_qa_chain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
# 6. MAIN
# ==============================
def main():
    print("🚀 Initializing system...\n")

    db = get_vector_db()
    llm = load_llm()
    qa_chain = create_qa_chain(llm, db)

    print("\n✅ System Ready! Ask your questions (type 'exit' to quit)\n")

    while True:
        query = input("🧑 Question: ").strip()

        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        if not query:
            continue

        try:
            response = qa_chain.invoke(query)
            print("\n🤖 Answer:", response)
        except Exception as e:
            print("\n❌ Error:", str(e))

        print("\n" + "=" * 60 + "\n")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
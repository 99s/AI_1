import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback

# ==============================
# 1. CONFIG
# ==============================
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Please set your HuggingFace token.")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

MODEL_ID = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# 2. LOAD LLM
# ==============================
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )

# ==============================
# 3. LOAD OR CREATE VECTOR DB
# ==============================
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    return db

# ==============================
# 4. PROMPT
# ==============================
def get_prompt():
    template = """
Use ONLY the provided context to answer the question.
If you do not know the answer, say "I don't know".
Do not make up answers.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# ==============================
# 5. CREATE QA PIPELINE
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
    db = get_vector_db()
    llm = load_llm()
    qa_chain = create_qa_chain(llm, db)

    print("System Ready! Ask your questions (type 'exit' to quit)")

    while True:
        query = input("Question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:
            response = qa_chain.invoke(query)
            print("Answer:", response)
        except Exception as e:
            print("FULL ERROR:")
            traceback.print_exc()

        print("-" * 50)

if __name__ == "__main__":
    main()
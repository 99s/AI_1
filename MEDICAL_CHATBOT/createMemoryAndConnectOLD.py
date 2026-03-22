import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
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

# huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# huggingface_repo_id = "google/flan-t5-large"
huggingface_repo_id = "HuggingFaceH4/zephyr-7b-beta"

# ==============================
# 2. LOAD LLM
# ==============================
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )

# ==============================
# 3. LOAD DOCUMENTS
# ==============================
def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# ==============================
# 4. SPLIT TEXT
# ==============================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# ==============================
# 5. CREATE VECTOR DB
# ==============================
def create_vector_db(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

# ==============================
# 6. LOAD VECTOR DB
# ==============================
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ==============================
# 7. PROMPT
# ==============================
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you dont know the answer, just say that you dont know.
Dont make up answers.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# ==============================
# 8. CREATE QA PIPELINE (LCEL)
# ==============================
def create_qa_chain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = set_custom_prompt()

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
# 9. MAIN
# ==============================
if __name__ == "__main__":

    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    texts = split_documents(documents)

    print("Creating vector DB...")
    db = create_vector_db(texts)

    # Optional: load instead of recreate
    # db = load_vector_db()

    print("Loading LLM...")
    llm = load_llm()

    print("Creating QA pipeline...")
    qa_chain = create_qa_chain(llm, db)

    print("\n✅ System Ready! Ask your questions (type 'exit' to quit)\n")

    while True:
        query = input("Question: ")

        if query.lower() == "exit":
            break

        response = qa_chain.invoke(query)

        print("\nAnswer:", response)
        print("\n" + "-" * 50 + "\n")
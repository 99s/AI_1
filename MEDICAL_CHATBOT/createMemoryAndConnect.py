import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ==============================
# 1. CONFIG
# ==============================
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Please set your HuggingFace token.")

DATA_PATH = "data/"  # folder where PDFs are stored
DB_FAISS_PATH = "vectorstore/db_faiss"

huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# ==============================
# 2. LOAD LLM
# ==============================
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )
    return llm

# ==============================
# 3. LOAD DATA
# ==============================
def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# ==============================
# 4. SPLIT TEXT
# ==============================
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    return texts

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

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# ==============================
# 7. PROMPT
# ==============================
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you dont know the answer, just say that you dont know.
Dont make up answers.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# ==============================
# 8. CREATE QA CHAIN
# ==============================
def create_qa_chain(llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )
    return qa_chain

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

    print("Loading LLM...")
    llm = load_llm()

    print("Creating QA chain...")
    qa_chain = create_qa_chain(llm, db)

    print("\n✅ System Ready! Ask your questions (type 'exit' to quit)\n")

    while True:
        query = input("Question: ")

        if query.lower() == "exit":
            break

        response = qa_chain({"query": query})

        print("\nAnswer:", response["result"])
        print("\n" + "-" * 50 + "\n")
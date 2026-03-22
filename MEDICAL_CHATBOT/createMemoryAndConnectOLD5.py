import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==============================
# 1. CONFIG
# ==============================
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

MODEL_ID = "gpt2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# 2. LOAD LLM (LOCAL GPT2)
# ==============================
def load_llm():
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )
   
    return HuggingFacePipeline(pipeline=pipe)

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
Context:
{context}

Question: {question}

Answer briefly:
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
# 6. MAIN
# ==============================
def main():
    print("Initializing system...")

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

            if not response or response.strip() == "":
                response = "No answer generated. Check your documents."

            print("Answer:", response)

        except Exception as e:
            import traceback
            traceback.print_exc()

        print("-" * 50)

if __name__ == "__main__":
    main()
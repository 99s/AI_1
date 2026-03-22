import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ==============================
# 1. CONFIG
# ==============================
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Use OpenAI Embeddings
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# ==============================
# 2. LOAD LLM
# ==============================
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# ==============================
# 3. LOAD OR CREATE VECTOR DB
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, EMBEDDINGS)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    return db

# ==============================
# 4. PROMPT
# ==============================
def get_prompt():
    template = """
    You are a medical assistant chatbot.

    Use ONLY the information provided in the context below.

    If the answer is NOT present in the context, say:
    "I don't know. Please ask a question related to the provided documents."

    DO NOT use your own knowledge.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
# def get_prompt():
#     template = """
# Context:
# {context}

# Question: {question}

# Answer briefly:
# """
#     return PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

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
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_vector_db():
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, EMBEDDINGS)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    return db

def get_prompt():
    template = """
Use ONLY context.
If not found say "I don't know".

Context:
{context}

Question: {question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_chain():
    db = get_vector_db()
    llm = load_llm()
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
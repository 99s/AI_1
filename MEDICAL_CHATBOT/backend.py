import os
from typing import List, Any

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever

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
# RRF RERANKING
# ==============================
def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    fused_scores = {}

    for docs_list in results:
        for rank, doc in enumerate(docs_list):
            doc_str = doc.model_dump_json()

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = []
    for doc_str, score in sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        reconstructed_doc = Document.model_validate_json(doc_str)
        reranked_results.append((reconstructed_doc, score))

    return reranked_results

# ==============================
# CUSTOM RETRIEVER (RRF)
# ==============================
class RerankRetriever(BaseRetriever):
    db: Any
    rrf_fn: Any

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs1 = self.db.similarity_search(query, k=5)
        docs2 = self.db.max_marginal_relevance_search(query, k=5, fetch_k=10)

        reranked = self.rrf_fn([docs1, docs2])

        final_docs = []
        for doc, score in reranked[:3]:
            doc.metadata["score"] = score
            final_docs.append(doc)

        return final_docs

# ==============================
# PROMPT
# ==============================
def get_prompt():
    template = """
You are a medical assistant chatbot.

Use ONLY the context below.
If answer not found, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# ==============================
# CREATE CHAIN
# ==============================
def create_chain():
    db = get_vector_db()
    llm = load_llm()
    prompt = get_prompt()

    # Create RRF retriever
    retriever = RerankRetriever(
        db=db,
        rrf_fn=reciprocal_rank_fusion
    )

    # Format docs + include scores
    def format_docs(docs):
        if not docs:
            return "NO_CONTEXT"

        formatted = []
        for doc in docs:
            score = doc.metadata.get("score", 0)
            content = doc.page_content[:500]

            formatted.append(
                f"[Score: {score:.4f}]\n{content}"
            )

        return "\n\n".join(formatted)

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
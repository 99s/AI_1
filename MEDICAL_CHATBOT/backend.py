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
from sentence_transformers import CrossEncoder

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
# CUSTOM RETRIEVER (FULL SCORING)
# ==============================
class RerankRetriever(BaseRetriever):
    db: Any
    rrf_fn: Any

    def _get_relevant_documents(self, query: str) -> List[Document]:

        # ------------------------------
        # 1. Similarity search (with score)
        # ------------------------------
        sim_results = self.db.similarity_search_with_score(query, k=5)

        docs1 = []
        sim_map = {}

        for doc, score in sim_results:
            key = doc.page_content
            docs1.append(doc)

            # Convert distance → similarity
            sim_map[key] = 1 / (1 + score)

        # ------------------------------
        # 2. MMR search
        # ------------------------------
        docs2 = self.db.max_marginal_relevance_search(query, k=5, fetch_k=10)

        # ------------------------------
        # 3. RRF
        # ------------------------------
        rrf_results = self.rrf_fn([docs1, docs2])

        rrf_map = {doc.page_content: score for doc, score in rrf_results}

        # ------------------------------
        # 4. Cross Encoder
        # ------------------------------
        pairs = [(query, doc.page_content) for doc, _ in rrf_results]
        ce_scores = cross_encoder.predict(pairs)

        # ------------------------------
        # 5. Combine scores
        # ------------------------------
        final_docs = []

        for (doc, _), ce_score in zip(rrf_results, ce_scores):

            key = doc.page_content

            sim = sim_map.get(key, 0.0)
            rrf = rrf_map.get(key, 0.0)
            ce = float(ce_score)

            # Final weighted score
            final = (
                0.4 * sim +
                0.3 * rrf +
                0.3 * ce
            )

            doc.metadata["sim_score"] = sim
            doc.metadata["rrf_score"] = rrf
            doc.metadata["ce_score"] = ce
            doc.metadata["final_score"] = final

            final_docs.append(doc)

        # Sort by final score
        final_docs.sort(
            key=lambda d: d.metadata["final_score"],
            reverse=True
        )

        return final_docs[:3]

# ==============================
# PROMPT
# ==============================
def get_prompt():
    template = """
You are a medical assistant chatbot.

Use the provided context to answer the question.

- If the context contains relevant information, answer using it.
- You may summarize or rephrase the context.
- If the context is partially relevant, still try to give a helpful answer.
- Only say "I don't know" if the context contains absolutely no useful information.

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
# FORMAT DOCS
# ==============================
def format_docs(docs):
    if not docs:
        return "NO_CONTEXT"

    formatted = []

    for i, doc in enumerate(docs, 1):
        sim = doc.metadata.get("sim_score", 0.0)
        rrf = doc.metadata.get("rrf_score", 0.0)
        ce = doc.metadata.get("ce_score", 0.0)
        final = doc.metadata.get("final_score", 0.0)

        content = doc.page_content[:500]

        formatted.append(
            f"[Doc {i}]\n"
            f"Final: {final:.4f} | Sim: {sim:.4f} | RRF: {rrf:.4f} | CE: {ce:.4f}\n"
            f"{content}"
        )

    return "\n\n".join(formatted)

# ==============================
# CREATE CHAIN
# ==============================
def create_chain():
    db = get_vector_db()
    llm = load_llm()
    prompt = get_prompt()

    retriever = RerankRetriever(
        db=db,
        rrf_fn=reciprocal_rank_fusion
    )

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
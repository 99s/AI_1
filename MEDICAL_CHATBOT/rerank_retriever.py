from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from langchain_core.documents import Document

class RerankRetriever(BaseRetriever):
    db: Any
    rrf_fn: Any

    def _get_relevant_documents(self, query: str) -> List[Document]:

        # ==============================
        # 1. Similarity Search WITH score
        # ==============================
        sim_results = self.db.similarity_search_with_score(query, k=5)

        docs1 = []
        sim_score_map = {}

        for doc, score in sim_results:
            doc_id = doc.page_content
            sim_score_map[doc_id] = float(score)
            docs1.append(doc)

        # ==============================
        # 2. MMR Search
        # ==============================
        docs2 = self.db.max_marginal_relevance_search(query, k=5, fetch_k=10)

        # ==============================
        # 3. RRF
        # ==============================
        rrf_results = self.rrf_fn([docs1, docs2])

        # Map RRF score
        rrf_score_map = {
            doc.page_content: score for doc, score in rrf_results
        }

        # ==============================
        # 4. Cross Encoder scoring
        # ==============================
        pairs = [(query, doc.page_content) for doc, _ in rrf_results]
        ce_scores = cross_encoder.predict(pairs)

        # ==============================
        # 5. Combine everything
        # ==============================
        final_docs = []

        for (doc, _), ce_score in zip(rrf_results, ce_scores):

            doc_id = doc.page_content

            sim_score = sim_score_map.get(doc_id, 0)
            rrf_score = rrf_score_map.get(doc_id, 0)

            # Normalize similarity (optional but recommended)
            sim_score = 1 / (1 + sim_score)

            # Final score (tunable weights)
            final_score = (
                0.4 * sim_score +
                0.3 * rrf_score +
                0.3 * ce_score
            )

            doc.metadata["sim_score"] = sim_score
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["ce_score"] = float(ce_score)
            doc.metadata["final_score"] = float(final_score)

            final_docs.append(doc)

        # Sort by final score
        final_docs.sort(
            key=lambda d: d.metadata["final_score"],
            reverse=True
        )

        return final_docs[:3]
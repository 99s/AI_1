from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from langchain_core.documents import Document

class RerankRetriever(BaseRetriever):
    db: Any
    rrf_fn: Any

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs1 = self.db.similarity_search(query, k=5)
        docs2 = self.db.max_marginal_relevance_search(query, k=5, fetch_k=10)

        reranked = self.rrf_fn([docs1, docs2])

        return reranked[:3]
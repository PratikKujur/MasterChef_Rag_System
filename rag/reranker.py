from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest


class Reranker:
    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        if cache_dir:
            self._ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        else:
            self._ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
    ) -> List[Document]:
        if not documents:
            return []

        passages = [
            {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata,
            }
            for idx, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        reranked_docs = []
        for item in results[:top_k]:
            doc = Document(
                page_content=item["text"],
                metadata=item.get("meta", {}),
            )
            reranked_docs.append(doc)

        return reranked_docs


class EmbeddingReranker(Reranker):
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, cache_dir=cache_dir)
        self.embedding_model = embedding_model

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
    ) -> List[tuple[Document, float]]:
        if not documents:
            return []

        passages = [
            {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata,
            }
            for idx, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        reranked = []
        for item in results[:top_k]:
            doc = Document(
                page_content=item["text"],
                metadata=item.get("meta", {}),
            )
            score = item.get("score", 0.0)
            reranked.append((doc, score))

        return reranked

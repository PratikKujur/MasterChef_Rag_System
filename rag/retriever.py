from typing import List, Optional, Union
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from vectorDb.vectorDb import SQLiteVecDB
from .reranker import Reranker, EmbeddingReranker


class BaseRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[Document]:
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: SQLiteVecDB,
        embedding_model: SentenceTransformer,
        default_top_k: int = 10,
        reranker_model: Optional[str] = "ms-marco-MiniLM-L-12-v2",
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k
        self._reranker: Optional[Reranker] = None
        if reranker_model:
            self._reranker = Reranker(model_name=reranker_model)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[Document]:
        retrieval_k = top_k * 3 if rerank and rerank_top_k else top_k

        documents = self.vector_store.similarity_search(
            query=query,
            k=retrieval_k,
            embedding_model=self.embedding_model,
        )

        if not rerank or not self._reranker:
            return documents[:top_k]

        actual_rerank_k = rerank_top_k if rerank_top_k else top_k
        reranked = self._reranker.rerank(
            query=query,
            documents=documents,
            top_k=actual_rerank_k,
        )
        return reranked

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[tuple[Document, float]]:
        documents = self.retrieve(
            query=query,
            top_k=top_k,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
        )

        scores = []
        query_embedding = self.embedding_model.encode([query])[0]
        for doc in documents:
            doc_embedding = self.embedding_model.encode([doc.page_content])[0]
            import numpy as np
            similarity = float(np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            ))
            scores.append((doc, similarity))

        return scores


class EnsembleRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> List[Document]:
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.retrieve(query, top_k=top_k * 2, rerank=False)
            for doc in docs:
                all_docs.append((doc, weight))

        doc_scores: dict[str, tuple[Document, float]] = {}
        for doc, weight in all_docs:
            key = doc.page_content[:100]
            if key not in doc_scores:
                doc_scores[key] = (doc, 0.0)
            doc_scores[key] = (doc, doc_scores[key][1] + weight)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]


def create_retriever(
    vector_store_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    reranker_model: Optional[str] = "ms-marco-MiniLM-L-12-v2",
    default_top_k: int = 10,
) -> VectorRetriever:
    embedding_model = SentenceTransformer(embedding_model_name)
    vector_store = SQLiteVecDB.load(vector_store_path, embedding_model)
    return VectorRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        default_top_k=default_top_k,
        reranker_model=reranker_model,
    )

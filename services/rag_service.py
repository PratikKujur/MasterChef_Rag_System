import os
import time
from typing import List
from langchain_core.documents import Document
from collections import deque

from offline_ingestion.pipeline import OfflineIngestionPipeline, load_vector_store
from rag.retriever import VectorRetriever
from rag.cache import query_cache

from core.config import DATA_PATH, VECTOR_STORE_PATH


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
VECTOR_STORE_EXISTS = os.path.exists(os.path.join(VECTOR_STORE_PATH, "vectors.db"))
HISTORY_SIZE = 3


def init_vector_store(force_recreate: bool = False) -> None:
    if VECTOR_STORE_EXISTS and not force_recreate:
        print(f"Vector store already exists at {VECTOR_STORE_PATH}")
        return
    
    print("Creating vector store...")
    if force_recreate and os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)
    
    pipeline = OfflineIngestionPipeline(
        file_path=DATA_PATH,
        chunk_size=10,
        buffer_size=1,
        embedder_name=EMBEDDING_MODEL,
        vector_store_path=str(VECTOR_STORE_PATH),
    )
    vector_store = pipeline.run()
    print(f"Vector store created with {vector_store.count()} documents")


def init_retriever() -> VectorRetriever:
    vector_store = load_vector_store(
        path=str(VECTOR_STORE_PATH),
        embedder_name=EMBEDDING_MODEL,
    )
    
    embedding_model = vector_store._embedding_model
    if embedding_model is None:
        raise ValueError("Embedding model not loaded properly")
    
    retriever = VectorRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        default_top_k=10,
        reranker_model=RERANKER_MODEL,
    )
    return retriever


def format_context(documents: List[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Source {i}] {doc.page_content} (Page: {page})")
    return "\n\n".join(context_parts)


class RAGService:
    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever
        self.conversation_histories: dict[str, deque] = {}
    
    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Document]:
        return self.retriever.retrieve(query=query, top_k=top_k, rerank=rerank)
    
    def add_to_history(self, session_id: str, query: str, answer: str) -> None:
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = deque(maxlen=HISTORY_SIZE)
        
        history = self.conversation_histories[session_id]
        if len(history) >= HISTORY_SIZE:
            history.popleft()
        history.append((query, answer))
    
    def get_history(self, session_id: str) -> deque:
        return self.conversation_histories.get(session_id, deque(maxlen=HISTORY_SIZE))
    
    def answer_query(
        self,
        query: str,
        chain,
        summarize_func,
        session_id: str = "default",
        use_rerank: bool = True,
        retrieval_k: int = 5,
        use_cache: bool = True,
    ) -> dict:
        t_start = time.time()
        
        if use_cache:
            cached = query_cache.get(query, retrieval_k, use_rerank)
            if cached:
                return {
                    "answer": cached.answer,
                    "documents": cached.documents,
                    "timing": {"total": time.time() - t_start, "cache_hit": True}
                }
        
        documents = self.retrieve(query, retrieval_k, use_rerank)
        context = format_context(documents)
        
        history = self.get_history(session_id)
        history_str = summarize_func(history)
        
        answer = chain.invoke({"context": context, "question": query, "history": history_str})
        
        self.add_to_history(session_id, query, answer)
        
        if use_cache:
            doc_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            query_cache.set(query, answer, doc_dicts, retrieval_k, use_rerank)
        
        return {
            "answer": answer,
            "documents": documents,
            "timing": {
                "total": time.time() - t_start,
                "retrieval": None,
                "llm": None,
                "cache_hit": True
            }
        }

import os
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from offline_ingestion.pipeline import OfflineIngestionPipeline, load_vector_store
from rag.retriever import VectorRetriever
from rag.reranker import Reranker
from rag.cache import query_cache, CacheEntry
from core.config import DATA_PATH, VECTOR_STORE_PATH


LLM_MODEL = "llama3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
VECTOR_STORE_EXISTS = os.path.exists(os.path.join(VECTOR_STORE_PATH, "vectors.db"))


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


def init_llm(model_name: str = LLM_MODEL) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.7,
        base_url="http://localhost:11434",
    )


def create_rag_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful cooking assistant. Use the following context from a cookbook to answer the user's question.
If you cannot find the answer in the context, say that you don't have that information.

Context:
{context}"""),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain


def format_context(documents: List[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Document {i}] (Source: {source}, Page: {page})\n{doc.page_content}")
    return "\n\n".join(context_parts)


def answer_query(
    query: str,
    retriever: VectorRetriever,
    llm: ChatOllama,
    use_rerank: bool = True,
    retrieval_k: int = 10,
    use_cache: bool = True,
) -> str:
    if use_cache:
        cached = query_cache.get(query, retrieval_k, use_rerank)
        if cached:
            print(f"\n--- Query: {query} ---")
            print("[CACHE HIT] Returning cached answer")
            print(f"Retrieved {len(cached.documents)} documents from cache")
            for i, doc in enumerate(cached.documents[:3], 1):
                preview = doc["page_content"][:100] + "..." if len(doc["page_content"]) > 100 else doc["page_content"]
                print(f"  {i}. {preview}")
            return cached.answer

    print(f"\n--- Query: {query} ---\n")
    
    documents = retriever.retrieve(
        query=query,
        top_k=retrieval_k,
        rerank=use_rerank,
    )
    
    print(f"Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  {i}. {preview}")
    
    context = format_context(documents)
    
    chain = create_rag_chain(llm)
    answer = chain.invoke({"context": context, "question": query})
    
    if use_cache:
        doc_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        query_cache.set(query, answer, doc_dicts, retrieval_k, use_rerank)
        print("[CACHE] Answer cached for 6 hours")
    
    return answer


def main():
    print("=" * 60)
    print("MasterChef RAG System")
    print("=" * 60)
    
    print("\n[Step 1] Initializing Vector Store...")
    init_vector_store(force_recreate=False)
    
    print("\n[Step 2] Loading Vector Retriever...")
    retriever = init_retriever()
    
    print("\n[Step 3] Initializing Ollama LLM...")
    llm = init_llm(LLM_MODEL)
    
    print("\n[Step 4] Testing Query (with caching)...")
    test_query = "What is a simple recipe for chicken?"
    answer = answer_query(
        query=test_query,
        retriever=retriever,
        llm=llm,
        use_rerank=True,
        retrieval_k=5,
        use_cache=True,
    )
    
    print(f"\n--- Answer ---\n{answer}")
    print("\n" + "=" * 60)
    print("RAG Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

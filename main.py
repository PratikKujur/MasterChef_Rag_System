import os
import time
from typing import List, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from collections import deque

from offline_ingestion.pipeline import OfflineIngestionPipeline, load_vector_store
from rag.retriever import VectorRetriever
from rag.reranker import Reranker
from rag.cache import query_cache, CacheEntry
from core.config import DATA_PATH, VECTOR_STORE_PATH


LLM_MODEL = "phi3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
VECTOR_STORE_EXISTS = os.path.exists(os.path.join(VECTOR_STORE_PATH, "vectors.db"))
HISTORY_SIZE = 3
MAX_HISTORY_SUMMARY_CHARS = 300


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
        temperature=0.2,
        base_url="http://localhost:11434",
        top_k=5,
        num_predict=100,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\n\n"],
        keep_alive=30,
        num_ctx=2048,    
    )


def create_rag_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful cooking assistant. Use the following context from a cookbook to answer the user's question.
If you cannot find the answer in the context, say that you don't have that information.
IMPORTANT: Always cite your sources by including the page number in brackets at the end of relevant statements. Example: "Add salt to taste [Page 5]"

{history}

Context:
{context}"""),
        ("human", "{question}"),
    ])
    print(f"Prompt created: {prompt}, prompt length: {len(prompt.format(context='[CONTEXT]', question='[QUESTION]', history='[HISTORY]'))} characters")
    chain = prompt | llm | StrOutputParser()
    return chain


def format_history(history: deque) -> str:
    if not history:
        return "No previous conversation history."
    
    formatted = ["Previous conversation:\n"]
    for i, (q, a) in enumerate(history, 1):
        formatted.append(f"Turn {i}:")
        formatted.append(f"  User: {q}")
        formatted.append(f"  Assistant: {a}\n")
    return "\n".join(formatted)


def summarize_history(history: deque, llm: Optional[ChatOllama] = None) -> str:
    if not history:
        return "No previous conversation history."
    
    raw_history = format_history(history)
    
    if llm is None:
        if len(raw_history) > MAX_HISTORY_SUMMARY_CHARS:
            return raw_history[:MAX_HISTORY_SUMMARY_CHARS] + "..."
        return raw_history
    
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You summarize conversation history briefly. Focus on key topics discussed and any constraints or preferences mentioned. Keep under 200 characters."),
        ("human", "Summarize this conversation:\n{history}"),
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    summary = summarize_chain.invoke({"history": raw_history})
    return f"Conversation summary: {summary}"


def format_context(documents: List[Document]) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Source {i}] {doc.page_content} (Page: {page})")
    return "\n\n".join(context_parts)


def answer_query(
    query: str,
    retriever: VectorRetriever,
    llm: ChatOllama,
    history: deque,
    use_rerank: bool = True,
    retrieval_k: int = 5,
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

    print(f"\n--- Query: {query} ---")
    
    t_start = time.time()
    t_retrieval = 0.0
    t_rerank = 0.0
    t_context = 0.0
    t_llm = 0.0
    
    documents = retriever.retrieve(
        query=query,
        top_k=retrieval_k,
        rerank=use_rerank,
    )
    t_retrieval = time.time() - t_start
    print(f"[TIMER] Retrieval: {t_retrieval:.2f}s")
    
    print(f"Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  {i}. {preview}")
    
    t_start = time.time()
    context = format_context(documents)
    history_str = summarize_history(history, llm)
    t_context = time.time() - t_start
    print(f"[TIMER] Context formatting: {t_context:.4f}s")
    print(f"[INFO] Context length: {len(context)} chars, History length: {len(history_str)} chars")
    
    chain = create_rag_chain(llm)
    t_start = time.time()
    answer = chain.invoke({"context": context, "question": query, "history": history_str})
    t_llm = time.time() - t_start
    print(f"[TIMER] LLM inference: {t_llm:.2f}s")
    
    print(f"[TIMER] Total time: {t_retrieval + t_context + t_llm:.2f}s")
    
    if len(history) >= HISTORY_SIZE:
        history.popleft()
    history.append((query, answer))
    
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
    t_start = time.time()
    init_vector_store(force_recreate=False)
    print(f"[TIMER] Vector store init: {time.time() - t_start:.2f}s")
    
    print("\n[Step 2] Loading Vector Retriever...")
    t_start = time.time()
    retriever = init_retriever()
    print(f"[TIMER] Retriever init: {time.time() - t_start:.2f}s")
    
    print("\n[Step 3] Initializing Ollama LLM...")
    t_start = time.time()
    llm = init_llm(LLM_MODEL)
    print(f"[TIMER] LLM init: {time.time() - t_start:.2f}s")
    
    conversation_history = deque(maxlen=HISTORY_SIZE)
    
    print("\n[Step 4] Testing Query (with caching)...")
    test_query = "What is a simple recipe for chicken?"
    answer = answer_query(
        query=test_query,
        retriever=retriever,
        llm=llm,
        history=conversation_history,
        use_rerank=True,
        retrieval_k=5,
        use_cache=False,
    )
    
    print(f"\n--- Answer ---\n{answer}")
    
    test_query2 = "Can I use turkey instead?"
    answer2 = answer_query(
        query=test_query2,
        retriever=retriever,
        llm=llm,
        history=conversation_history,
        use_rerank=True,
        retrieval_k=5,
        use_cache=False,
    )
    print(f"\n--- Answer ---\n{answer2}")
    
    test_query3 = "What spices do I need?"
    answer3 = answer_query(
        query=test_query3,
        retriever=retriever,
        llm=llm,
        history=conversation_history,
        use_rerank=True,
        retrieval_k=5,
        use_cache=False,
    )
    print(f"\n--- Answer ---\n{answer3}")
    
    print("\n--- Conversation History Summary (last 3) ---")
    print(summarize_history(conversation_history, llm))
    
    print("\n" + "=" * 60)
    print("RAG Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

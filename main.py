from services.llm_service import init_llm, create_rag_chain, summarize_history
from services.rag_service import init_vector_store, init_retriever, RAGService


def main():
    print("=" * 60)
    print("MasterChef RAG System")
    print("=" * 60)
    
    print("\n[Step 1] Initializing Vector Store...")
    init_vector_store(force_recreate=False)
    
    print("\n[Step 2] Loading Vector Retriever...")
    retriever = init_retriever()
    
    print("\n[Step 3] Initializing Ollama LLM...")
    llm = init_llm()
    chain = create_rag_chain(llm)
    
    rag = RAGService(retriever)
    session_id = "default"
    
    queries = [
        "What is a simple recipe for chicken?",
        "Can I use turkey instead?",
        "What spices do I need?",
    ]
    
    print("\n[Step 4] Processing queries...")
    for query in queries:
        print(f"\n--- Query: {query} ---")
        result = rag.answer_query(
            query=query,
            chain=chain,
            summarize_func=lambda h: summarize_history(h, llm),
            session_id=session_id,
            use_rerank=True,
            retrieval_k=5,
            use_cache=False,
        )
        print(f"Answer: {result['answer']}")
        print(f"Timing: {result['timing']['total']:.2f}s")
    
    print("\n" + "=" * 60)
    print("RAG Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

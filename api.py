from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from collections import deque
import uuid

from services.rag_service import init_vector_store, init_retriever, RAGService
from services.llm_service import init_llm, create_rag_chain, summarize_history


app = FastAPI(title="MasterChef RAG API", version="1.0.0")

retriever = None
llm = None
rag_service = None


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_rerank: bool = True
    retrieval_k: int = 10
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    timing: dict

class HistoryResponse(BaseModel):
    session_id: str
    history: List[dict]


@app.on_event("startup")
async def startup_event():
    global retriever, llm, rag_service
    print("Initializing RAG system...")
    init_vector_store(force_recreate=False)
    retriever = init_retriever()
    llm = init_llm()
    rag_service = RAGService(retriever)
    print("RAG system ready!")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if rag_service is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    session_id = request.session_id or str(uuid.uuid4())
    chain = create_rag_chain(llm)
    
    result = rag_service.answer_query(
        query=request.query,
        chain=chain,
        summarize_func=lambda h: summarize_history(h, llm),
        session_id=session_id,
        use_rerank=request.use_rerank,
        retrieval_k=request.retrieval_k,
        use_cache=request.use_cache,
    )
    
    sources = []
    for doc in result["documents"]:
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A")
        })
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        timing={"total": result["timing"]["total"], "cache_hit": result["timing"]["cache_hit"]}
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    history = rag_service.get_history(session_id) if rag_service else deque()
    
    history_list = [
        {"question": q, "answer": a}
        for q, a in history
    ]
    
    return HistoryResponse(session_id=session_id, history=history_list)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    if rag_service and session_id in rag_service.conversation_histories:
        rag_service.conversation_histories[session_id].clear()
    return {"message": "History cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    if rag_service:
        return {
            "active_sessions": len(rag_service.conversation_histories),
            "session_ids": list(rag_service.conversation_histories.keys())
        }
    return {"active_sessions": 0, "session_ids": []}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_service_ready": rag_service is not None,
        "llm_ready": llm is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

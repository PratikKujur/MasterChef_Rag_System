from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from collections import deque
import uuid

from main import init_vector_store, init_retriever, init_llm, answer_query, format_context
from core.config import DATA_PATH, VECTOR_STORE_PATH


app = FastAPI(title="MasterChef RAG API", version="1.0.0")

retriever = None
llm = None
conversation_histories: dict[str, deque] = {}

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

class HistoryResponse(BaseModel):
    session_id: str
    history: List[dict]


@app.on_event("startup")
async def startup_event():
    global retriever, llm
    print("Initializing RAG system...")
    init_vector_store(force_recreate=False)
    retriever = init_retriever()
    llm = init_llm()
    print("RAG system ready!")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in conversation_histories:
        conversation_histories[session_id] = deque(maxlen=3)
    
    history = conversation_histories[session_id]
    
    answer = answer_query(
        query=request.query,
        retriever=retriever,
        llm=llm,
        history=history,
        use_rerank=request.use_rerank,
        retrieval_k=request.retrieval_k,
        use_cache=request.use_cache,
    )
    
    sources = []
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=session_id
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    if session_id not in conversation_histories:
        return HistoryResponse(session_id=session_id, history=[])
    
    history_list = [
        {"question": q, "answer": a}
        for q, a in conversation_histories[session_id]
    ]
    
    return HistoryResponse(session_id=session_id, history=history_list)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    if session_id in conversation_histories:
        conversation_histories[session_id].clear()
    return {"message": "History cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    return {
        "active_sessions": len(conversation_histories),
        "session_ids": list(conversation_histories.keys())
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retriever_ready": retriever is not None,
        "llm_ready": llm is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict


CACHE_DIR = os.path.join(os.getcwd(), "vectorDb", "cache")
CACHE_TTL_SECONDS = 6 * 60 * 60


@dataclass
class CacheEntry:
    query: str
    answer: str
    documents: list
    timestamp: float
    retrieval_k: int
    use_rerank: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "documents": [doc if isinstance(doc, dict) else {"page_content": doc.page_content, "metadata": doc.metadata} for doc in self.documents],
            "timestamp": self.timestamp,
            "retrieval_k": self.retrieval_k,
            "use_rerank": self.use_rerank,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(
            query=data["query"],
            answer=data["answer"],
            documents=data["documents"],
            timestamp=data["timestamp"],
            retrieval_k=data["retrieval_k"],
            use_rerank=data["use_rerank"],
        )


def _get_cache_key(query: str, retrieval_k: int, use_rerank: bool) -> str:
    raw = f"{query.lower().strip()}|{retrieval_k}|{use_rerank}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _get_cache_path(cache_key: str) -> Path:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return Path(CACHE_DIR) / f"{cache_key}.json"


class QueryCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds

    def get(self, query: str, retrieval_k: int = 10, use_rerank: bool = True) -> Optional[CacheEntry]:
        cache_key = _get_cache_key(query, retrieval_k, use_rerank)
        cache_path = _get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            age = time.time() - entry.timestamp
            
            if age > self.ttl:
                cache_path.unlink(missing_ok=True)
                return None
            
            return entry
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def set(
        self,
        query: str,
        answer: str,
        documents: list,
        retrieval_k: int = 10,
        use_rerank: bool = True,
    ) -> None:
        cache_key = _get_cache_key(query, retrieval_k, use_rerank)
        cache_path = _get_cache_path(cache_key)
        
        entry = CacheEntry(
            query=query,
            answer=answer,
            documents=documents,
            timestamp=time.time(),
            retrieval_k=retrieval_k,
            use_rerank=use_rerank,
        )
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

    def clear(self) -> int:
        count = 0
        for path in Path(CACHE_DIR).glob("*.json"):
            path.unlink(missing_ok=True)
            count += 1
        return count

    def clear_expired(self) -> int:
        count = 0
        now = time.time()
        for path in Path(CACHE_DIR).glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if now - data["timestamp"] > self.ttl:
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError):
                path.unlink(missing_ok=True)
                count += 1
        return count


query_cache = QueryCache()

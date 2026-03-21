from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os
import json
import shutil
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import sqlite3


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[np.ndarray]] = None
    ) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, embedding_model: SentenceTransformer) -> "BaseVectorStore":
        pass


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [emb.tolist() for emb in self.model.encode(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def __call__(
        self, input: str | List[str]
    ) -> List[List[float]] | List[float]:
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(input)


class SQLiteVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        db_path: str = "vector_store.db",
        table_name: str = "vectors",
        distance_metric: str = "cosine",
    ):
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.table_name = table_name
        self.distance_metric = distance_metric
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            cursor = self._conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    vector BLOB
                )
            """)
            self._conn.commit()
        return self._conn

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self.distance_metric == "cosine":
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return 1 - (dot / (norm1 * norm2))
        elif self.distance_metric == "euclidean":
            return float(np.linalg.norm(vec1 - vec2))
        return float(np.dot(vec1 - vec2, vec1 - vec2))

    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[np.ndarray]] = None
    ) -> None:
        if embeddings is None:
            texts = [doc.page_content for doc in documents]
            emb_array = self.embedding_model.encode(texts)
            embeddings_list: List[np.ndarray] = list(emb_array)
        else:
            embeddings_list = embeddings

        conn = self._get_connection()
        cursor = conn.cursor()
        for doc, embedding in zip(documents, embeddings_list):
            metadata_json = json.dumps(doc.metadata)
            vector_bytes = np.asarray(embedding).astype(np.float32).tobytes()
            cursor.execute(
                f"INSERT INTO {self.table_name} (content, metadata, vector) VALUES (?, ?, ?)",
                (doc.page_content, metadata_json, vector_bytes),
            )
        conn.commit()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self.embedding_model.encode([query])[0]
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, content, metadata, vector FROM {self.table_name}")

        rows = cursor.fetchall()
        results: List[tuple[float, Any]] = []
        for row in rows:
            vector = np.frombuffer(row[3], dtype=np.float32)
            distance = self._compute_distance(query_embedding, vector)
            results.append((distance, row))

        results.sort(key=lambda x: x[0])
        top_k = results[:k]

        documents: List[Document] = []
        for distance, row in top_k:
            metadata = json.loads(row[2]) if row[2] else {}
            doc = Document(page_content=row[1], metadata=metadata)
            documents.append(doc)

        return documents

    def save(self, path: str) -> None:
        if self._conn and self.db_path != path:
            self._conn.close()
            shutil.copy2(self.db_path, path)
            self._conn = sqlite3.connect(path)

    @classmethod
    def load(cls, path: str, embedding_model: SentenceTransformer) -> "SQLiteVectorStore":
        return cls(embedding_model=embedding_model, db_path=path)


class VectorStoreFactory:
    _STORES: Dict[str, type] = {"sqlite": SQLiteVectorStore}

    @classmethod
    def create(
        cls, store_type: str, embedding_model: SentenceTransformer, **kwargs: Any
    ) -> BaseVectorStore:
        store_type_lower = store_type.lower()
        if store_type_lower not in cls._STORES:
            available = list(cls._STORES.keys())
            raise ValueError(
                f"Unsupported store type: {store_type}. Available: {available}"
            )
        return cls._STORES[store_type_lower](embedding_model=embedding_model, **kwargs)

    @classmethod
    def register(cls, name: str, store_class: type) -> None:
        cls._STORES[name.lower()] = store_class

    @classmethod
    def available_stores(cls) -> List[str]:
        return list(cls._STORES.keys())

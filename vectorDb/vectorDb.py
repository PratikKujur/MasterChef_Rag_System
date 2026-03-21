import os
import json
import shutil
from typing import List, Optional, Any
from pathlib import Path

import numpy as np
import sqlite3
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class SQLiteVecDB:
    def __init__(
        self,
        db_path: str,
        table_name: str = "vectors",
        distance_metric: str = "cosine",
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.distance_metric = distance_metric
        self._conn: Optional[sqlite3.Connection] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            cursor = self._conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created 
                ON {self.table_name}(created_at)
            """)
            self._conn.commit()
        return self._conn

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self.distance_metric == "cosine":
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1 - (dot / (norm1 * norm2))
        elif self.distance_metric == "euclidean":
            return float(np.linalg.norm(vec1 - vec2))
        return float(np.dot(vec1 - vec2, vec1 - vec2))

    def set_embedding_model(self, model: SentenceTransformer) -> None:
        self._embedding_model = model

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> None:
        if not documents:
            return

        if embeddings is None:
            if self._embedding_model is None:
                raise ValueError("Embedding model not set. Call set_embedding_model first.")
            texts = [doc.page_content for doc in documents]
            embeddings = list(self._embedding_model.encode(texts))

        conn = self._get_connection()
        cursor = conn.cursor()
        for doc, embedding in zip(documents, embeddings):
            metadata_json = json.dumps(doc.metadata)
            vector_bytes = np.asarray(embedding).astype(np.float32).tobytes()
            cursor.execute(
                f"INSERT INTO {self.table_name} (content, metadata, vector) VALUES (?, ?, ?)",
                (doc.page_content, metadata_json, vector_bytes),
            )
        conn.commit()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> List[Document]:
        model = embedding_model or self._embedding_model
        if model is None:
            raise ValueError("Embedding model not set.")

        query_embedding = model.encode([query])[0]
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, content, metadata, vector FROM {self.table_name}")

        rows = cursor.fetchall()
        if not rows:
            return []

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
        dest_path = Path(path) / "vectors.db"
        os.makedirs(path, exist_ok=True)
        
        if Path(self.db_path).resolve() == dest_path.resolve():
            return
        
        if self._conn:
            self._conn.close()
            self._conn = None
        shutil.copy2(self.db_path, dest_path)

    @classmethod
    def load(cls, path: str, embedding_model: SentenceTransformer) -> "SQLiteVecDB":
        db_path = Path(path) / "vectors.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Vector database not found at: {db_path}")
        instance = cls(db_path=str(db_path))
        instance.set_embedding_model(embedding_model)
        return instance

    def count(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def delete_all(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name}")
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

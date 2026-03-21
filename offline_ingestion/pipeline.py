import os
from typing import Optional, List
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import numpy as np

from .loader import DataLoader
from .chunker import DataChunker
from vectorDb.vectorDb import SQLiteVecDB
from core.config import DATA_PATH, VECTOR_STORE_PATH


class OfflineIngestionPipeline:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 10,
        buffer_size: int = 1,
        embedder_name: str = "all-MiniLM-L6-v2",
        vector_store_path: Optional[str] = None,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.embedder_name = embedder_name
        self.vector_store_path = vector_store_path or str(VECTOR_STORE_PATH)
        self.loader = None
        self.chunker = None
        self.vector_store: Optional[SQLiteVecDB] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    def run(self) -> SQLiteVecDB:
        data = self._load()
        chunks = self._chunk(data)
        documents, embeddings = self._embed(chunks)
        self._store(documents, embeddings)
        assert self.vector_store is not None
        return self.vector_store

    def _load(self) -> List[Document]:
        self.loader = DataLoader(self.file_path)
        return self.loader.load_data()

    def _chunk(self, data: List[Document]) -> List[Document]:
        self.chunker = DataChunker(
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size,
            embedder_name=self.embedder_name,
            data=data,
        )
        return self.chunker.chunk_data()

    def _embed(self, chunks: List[Document]) -> tuple[List[Document], List[np.ndarray]]:
        self._embedding_model = SentenceTransformer(self.embedder_name)
        texts = [chunk.page_content for chunk in chunks]
        embeddings: List[np.ndarray] = list(self._embedding_model.encode(texts))
        return chunks, embeddings

    def _store(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        assert self._embedding_model is not None
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        db_path = str(Path(self.vector_store_path) / "vectors.db")
        self.vector_store = SQLiteVecDB(db_path=db_path)
        self.vector_store.set_embedding_model(self._embedding_model)
        self.vector_store.add_documents(documents, embeddings)
        self.vector_store.save(self.vector_store_path)


def load_vector_store(
    path: Optional[str] = None,
    embedder_name: str = "all-MiniLM-L6-v2",
) -> SQLiteVecDB:
    path = path or str(VECTOR_STORE_PATH)
    embedding_model = SentenceTransformer(embedder_name)
    return SQLiteVecDB.load(path, embedding_model)


if __name__ == "__main__":
    import os
    
    pipeline = OfflineIngestionPipeline(
        file_path=DATA_PATH,
        chunk_size=10,
        buffer_size=1,
        embedder_name="all-MiniLM-L6-v2",
        vector_store_path=str(VECTOR_STORE_PATH),
    )
    vector_store = pipeline.run()
    print(f"Vector store created at: {pipeline.vector_store_path}")

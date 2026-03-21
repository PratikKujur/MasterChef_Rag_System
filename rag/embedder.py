from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder:
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )


class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.embedder = SentenceTransformerEmbedder(
            model_name=model_name, batch_size=batch_size, device=device
        )

    def _extract_texts(self, chunks: List[dict]) -> List[str]:
        return [chunk.get("page_content", "") for chunk in chunks]

    def generate_embeddings(
        self, chunks: List[dict], include_metadata: bool = True
    ) -> List[dict]:
        texts = self._extract_texts(chunks)
        embeddings = self.embedder.embed(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedder.embed(texts)
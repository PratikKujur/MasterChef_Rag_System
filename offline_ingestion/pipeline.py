from .loader import DataLoader
from .chunker import DataChunker
from .embed import Embedder
from core.config import DATA_PATH


class OfflineIngestionPipeline:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 10,
        buffer_size: int = 1,
        embedder_name: str = "all-MiniLM-L6-v2"
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.embedder_name = embedder_name
        self.loader = None
        self.chunker = None
        self.embedder = None

    def run(self):
        data = self._load()
        chunks = self._chunk(data)
        embedded_chunks = self._embed(chunks)
        return embedded_chunks

    def _load(self):
        self.loader = DataLoader(self.file_path)
        return self.loader.load_data()

    def _chunk(self, data):
        self.chunker = DataChunker(
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size,
            embedder_name=self.embedder_name,
            data=data
        )
        return self.chunker.chunk_data()

    def _embed(self, chunks):
        self.embedder = Embedder(self.embedder_name)
        chunk_dicts = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
        return self.embedder.generate_embeddings(chunk_dicts)


if __name__ == "__main__":
    pipeline = OfflineIngestionPipeline(
        file_path=DATA_PATH,
        chunk_size=10,
        buffer_size=1,
        embedder_name="all-MiniLM-L6-v2"
    )
    results = pipeline.run()
    print(f"Processed {len(results)} chunks")

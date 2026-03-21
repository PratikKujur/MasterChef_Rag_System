from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker as LangSemanticChunker
from sentence_transformers import SentenceTransformer


class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass


class TextCleaner(BasePreprocessor):
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        expand_contractions: bool = True,
        remove_stopwords: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self._stop_words: Optional[set] = None
        self._lemmatizer: Optional[WordNetLemmatizer] = None

    def _ensure_nltk_resources(self):
        if self._stop_words is None:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            self._stop_words = set(stopwords.words("english"))
            self._lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        self._ensure_nltk_resources()
        text = BeautifulSoup(text, "html.parser").get_text()
        if self.expand_contractions:
            text = str(contractions.fix(text))
        if self.lowercase:
            text = text.lower()
        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        if self.remove_stopwords and self._stop_words:
            text = " ".join(
                word for word in text.split() if word not in self._stop_words
            )
        if self._lemmatizer:
            text = self._lemmatizer.lemmatize(text)
        return text


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass


class SemanticSplitter(BaseChunker):
    def __init__(
        self,
        embedder: SentenceTransformer,
        chunk_size: int = 10,
        buffer_size: int = 1,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self._splitter = LangSemanticChunker(
            embeddings=embedder,  # type: ignore
            buffer_size=buffer_size,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self._splitter.create_documents(texts, metadatas=metadatas)


class DataChunker:
    def __init__(
        self,
        data: List[Document],
        chunk_size: int = 10,
        buffer_size: int = 1,
        embedder_name: str = "all-MiniLM-L6-v2",
        preprocessor: Optional[BasePreprocessor] = None,
    ):
        self.data = data
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.embedder = SentenceTransformer(embedder_name)
        self.preprocessor = preprocessor or TextCleaner()

    def _preprocess_data(self) -> List[Document]:
        preprocessed = []
        for doc in self.data:
            cleaned_text = self.preprocessor.preprocess(doc.page_content)
            preprocessed.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )
        return preprocessed

    def chunk_data(self) -> List[Document]:
        clean_data = self._preprocess_data()
        chunker = SemanticSplitter(
            embedder=self.embedder,
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size,
        )
        return chunker.chunk(clean_data)

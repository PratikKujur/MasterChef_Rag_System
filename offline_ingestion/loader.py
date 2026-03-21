from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, CSVLoader as LangCSVLoader, TextLoader


class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        pass


class PDFLoader(BaseLoader):
    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        return loader.load()


class CSVLoader(BaseLoader):
    def load(self) -> List[Document]:
        loader = LangCSVLoader(self.file_path)
        return loader.load()


class TextFileLoader(BaseLoader):
    def load(self) -> List[Document]:
        loader = TextLoader(self.file_path)
        return loader.load()


class DataLoader:
    _LOADERS = {
        ".pdf": PDFLoader,
        ".csv": CSVLoader,
        ".txt": TextFileLoader,
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._loader: Optional[BaseLoader] = None

    def _get_loader(self) -> BaseLoader:
        ext = self.file_path.lower().split(".")[-1]
        if ext not in self._LOADERS:
            raise ValueError(f"Unsupported file type: .{ext}")
        return self._LOADERS[ext](self.file_path)

    def load_data(self) -> List[Document]:
        self._loader = self._get_loader()
        return self._loader.load()

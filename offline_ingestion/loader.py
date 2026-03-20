from langchain_community.document_loaders import PyPDFLoader


class DataLoader:
    def __init__(self,file_path):
        self.file_path=file_path
    
    def load_data(self):
        Loader=PyPDFLoader(self.file_path)
        data=Loader.load()
        return data
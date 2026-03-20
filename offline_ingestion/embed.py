from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self,Model_embedder_name:str=None):
        self.model=SentenceTransformer(Model_embedder_name)

    def generate_embeddings(self,chunk:list[dict]):
        texts=[item['page_content'] for item in chunk]
        embeddings=self.model.encode(texts)
        for i,item in enumerate(chunk):
            item['embedding']=embeddings[i]
        return chunk
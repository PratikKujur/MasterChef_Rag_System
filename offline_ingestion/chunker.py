import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer


class DataChunker:
    def __init__(self,chunk_size=None,buffer_size=None,Model_embedder_name:str=None,data:list[dict]|None=None):
        self.chunk_size=chunk_size
        self.buffer_size=buffer_size
        self.embedder=SentenceTransformer(Model_embedder_name)
        self.data=data
    
    def text_preprocessing(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        lammatizer=WordNetLemmatizer()
        stop_words=set(stopwords.words('english'))
        

        for item in self.data:
            text=item['page_content']
            text=BeautifulSoup(text, 'html.parser').get_text()
            text=contractions.fix(text)
            text=text.lower()
            text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text=' '.join([word for word in text.split() if word not in stop_words])
            text=lammatizer.lemmatize(text)
            item['page_content']=text
        return self.data

    def chunk_data(self):
        clean_data=self.text_preprocessing()
        clean_text=[item['page_content'] for item in clean_data]
        meta_data=[item['metadata'] for item in clean_data]
        splitter=SemanticChunker(number_of_chunks=self.chunk_size,buffer_size=self.buffer_size,embedder=self.embedder)
        chunks=splitter.create_documents(text=clean_text,metadatas=meta_data)
        return chunks

        

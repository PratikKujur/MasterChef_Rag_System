import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH=os.path.join(os.getcwd(),"offline_ingestion","RawData","MasterChef-Cookbook-pdf.pdf")
VECTOR_STORE_PATH=os.path.join(os.getcwd(),"vectorDb","VectorStore")
API_URL=os.getenv("API_URL", "http://localhost:8000")

TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", "")
groq_api_key = os.getenv("GROQ_API_KEY")

LLM_MODEL = "phi3"
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_HISTORY_SUMMARY_CHARS = 300

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
VECTOR_STORE_EXISTS = os.path.exists(os.path.join(VECTOR_STORE_PATH, "vectors.db"))
HISTORY_SIZE = 3
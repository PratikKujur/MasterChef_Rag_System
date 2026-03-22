# MasterChef RAG System

A Retrieval-Augmented Generation (RAG) system for answering cooking questions based on the MasterChef cookbook.

## Features

| Feature | Description |
|---------|-------------|
| **History Awareness** | Maintains conversation context using a 3-turn sliding window with LLM-based summarization for long conversations |
| **Source Citation** | Every answer includes page number citations from the cookbook, e.g., `[Page 5]` |
| **Embedding Cache** | Caches generated embeddings in SQLite to avoid recomputation on repeated queries |
| **Reranking** | Uses FlashRank to rerank retrieval results for improved answer quality |

## System Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Interfaces                                 │
├──────────────────┬──────────────────┬────────────────────────────────────────┤
│    Gradio UI     │  Telegram Bot    │         FastAPI Endpoint               │
│     (ui.py)      │    (bot.py)      │            (api.py)                    │
└────────┬─────────┴────────┬─────────┴────────────────┬───────────────────────┘
         │                  │                         │
         └──────────────────┼─────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         LLM Service Layer                                     │
│                     (services/llm_service.py)                                 │
│          ChatGroq (if GROQ_API_KEY) │ ChatOllama (fallback)                   │
└────────────────────────────┬──────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┬────────────────────────────┐
         │                   │                   │                            │
         ▼                   ▼                   │                            ▼
┌─────────────────┐ ┌───────────────┐ ┌──────────┴──────┐┌────────────────────┐
│     Cache       │ │   Retriever   │ │   History Mgr   │ │     Reranker      │
│   (rag/cache)   │ │(rag/retriever)│ │  (3-window)     │ │   (rag/reranker)  │
└────────┬────────┘ └───────┬───────┘ └─────────────────┘ └─────────┬─────────┘
         │                  │                                       │
         │                  │          ┌────────────────────────────┘
         │                  │          │
         │          ┌───────┴──────────┴──────────┐
         │          │                             │
         │          ▼                             ▼
         │ ┌─────────────────┐        ┌─────────────────────────┐
         └►│  Embedder       │        │   Vector Store          │
           │ (rag/embedder)  │        │ (FAISS + SQLite-Vec)    │
           └────────┬────────┘        └─────────────────────────┘
                    │
                    ▼
           ┌────────────────┐
           │  Embedding     │
           │    Cache       │
           │ (SQLite)       │
           └────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **ui.py** | Gradio web interface for conversational Q&A |
| **bot.py** | Telegram bot for messaging-based interaction |
| **api.py** | FastAPI server exposing REST endpoints |
| **main.py** | Main entry point for running the application |
| **services/llm_service.py** | LLM abstraction (ChatGroq/Ollama) |
| **services/rag_service.py** | Core RAG orchestration logic |
| **rag/cache.py** | Embedding cache for query results |
| **rag/embedder.py** | Sentence embedding generation |
| **rag/retriever.py** | Document retrieval from vector store |
| **rag/reranker.py** | FlashRank-based result reranking |
| **rag/vector_store.py** | Vector storage (FAISS + SQLite-Vec) |
| **vectorDb/** | Vector database management |
| **offline_ingestion/** | PDF processing and data ingestion pipeline |

## How to Run Locally

### Prerequisites

1. **Python 3.10+**
2. **Ollama** (optional, for local LLM, Note: Ollama letency depends on your hardware confriguration)
   - Install from [ollama.ai](https://ollama.ai)
   - Pull model: `ollama pull phi3`
3. **API Keys** (see options below)

### API Configuration

Create a `.env` file in the project root with the following variables:

```env
# Required for Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Required for ChatGroq (recommended - faster and better quality)
GROQ_API_KEY=your_groq_api_key
```

Get your Groq API key from [console.groq.com](https://console.groq.com/keys).

### Running Options

#### Option 1: Gradio UI (Recommended with Groq API)

```bash
python ui.py
```

This starts a web interface at `http://localhost:7860`. Best experience when `GROQ_API_KEY` is set.

#### Option 2: Telegram Bot

```bash
python bot.py
```

Requires valid `TELEGRAM_BOT_TOKEN` in `.env`.

#### Option 3: FastAPI Server

```bash
python main.py
# or
uvicorn api:app --reload
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

#### Option 4: Ollama Only (No API Keys)

If running without any API keys, the system will use local Ollama:

```bash
# Ensure Ollama is running
ollama serve

# Then run any of the above options
python ui.py
```

### Initial Data Ingestion

If this is the first run, ingest the cookbook data:

```bash
python -m offline_ingestion.pipeline
```

This processes `offline_ingestion/RawData/MasterChef-Cookbook-pdf.pdf` and creates the vector store.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | For bot only | - | Telegram bot token from @BotFather |
| `GROQ_API_KEY` | For Groq API | - | Get from console.groq.com |
| `API_URL` | No | http://localhost:8000 | API base URL |

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
MasterChef_Rag_System/
├── api.py                 # FastAPI endpoints
├── main.py                # Main application entry
├── ui.py                  # Gradio web UI
├── bot.py                 # Telegram bot
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── core/
│   └── config.py          # Configuration settings
├── services/
│   ├── llm_service.py      # LLM abstraction layer
│   └── rag_service.py     # RAG pipeline logic
├── rag/
│   ├── cache.py            # Embedding & result cache
│   ├── embedder.py         # Sentence embedding
│   ├── retriever.py        # Document retrieval
│   ├── reranker.py         # Result reranking
│   └── vector_store.py     # Vector storage
├── vectorDb/
│   └── vectorDb.py         # FAISS + SQLite-Vec
└── offline_ingestion/
    ├── pipeline.py         # Ingestion pipeline
    ├── loader.py           # PDF loading
    ├── chunker.py          # Text splitting
    └── embed.py            # Embedding generation
```

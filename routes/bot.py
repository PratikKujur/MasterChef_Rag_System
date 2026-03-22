import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
import requests

from core.config import TELEGRAM_BOT_TOKEN

API_URL = os.getenv("API_URL", "http://localhost:8000")
BOT_TOKEN = TELEGRAM_BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASKING = range(1)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👨‍🍳 *Welcome to MasterChef RAG Bot!*\n\n"
        "I can answer your cooking questions using the cookbook.\n\n"
        "Commands:\n"
        "/ask - Ask a cooking question\n"
        "/help - Get help\n"
        "/about - About this bot\n"
        "/history - View conversation history\n"
        "/clear - Clear conversation history\n\n"
        "Just send me your question or use /ask to start!",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use this bot*\n\n"
        "1. Type your cooking question directly\n"
        "2. Or use /ask followed by your question\n"
        "3. The bot will search the cookbook and give you an answer with page citations\n\n"
        "*Tips:*\n"
        "• Be specific about ingredients\n"
        "• Mention if you want alternatives\n"
        "• Ask follow-up questions for more details",
        parse_mode="Markdown"
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🍳 *MasterChef RAG Bot*\n\n"
        "A smart cooking assistant powered by:\n"
        "• RAG (Retrieval-Augmented Generation)\n"
        "• Ollama LLM (Local AI)\n"
        "• Vector Search for relevant recipes\n\n"
        "This bot searches through a cookbook to find "
        "relevant recipes and cooking tips, then generates "
        "helpful answers with source citations.\n\n"
        "Version: 1.0.0",
        parse_mode="Markdown"
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = " ".join(context.args) if context.args else ""
    
    if not user_question:
        await update.message.reply_text(
            "❓ *Ask a question*\n\n"
            "Usage: /ask <your question>\n\n"
            "Example: /ask What is a simple chicken recipe?",
            parse_mode="Markdown"
        )
        return ConversationHandler.END
    
    await update.message.reply_text("🔍 Searching cookbook...")
    
    try:
        session_id = str(update.effective_user.id)
        
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": user_question,
                "session_id": session_id,
                "use_rerank": True,
                "retrieval_k": 5,
                "use_cache": True
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "Sorry, no answer found.")
        timing = data.get("timing", {})
        
        message = f"🍳 *Answer:*\n\n{answer}\n\n"
        
        message += f"\n⏱️ Response time: {timing.get('total', 0):.1f}s"
        
        if timing.get("cache_hit"):
            message += " (cached)"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "❌ *Error:* Cannot connect to the RAG server.\n"
            "Please try again later."
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ *Error:* {str(e)}")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    
    try:
        response = requests.get(f"{API_URL}/history/{session_id}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        history = data.get("history", [])
        
        if not history:
            await update.message.reply_text("📭 No conversation history found.")
            return
        
        message = "📜 *Conversation History:*\n\n"
        for i, item in enumerate(history, 1):
            q = item.get("question", "")[:50]
            a = item.get("answer", "")[:100]
            message += f"*Turn {i}:*\n"
            message += f"Q: {q}...\n"
            message += f"A: {a}...\n\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Error fetching history.")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_id = str(update.effective_user.id)
    
    try:
        response = requests.delete(f"{API_URL}/history/{session_id}", timeout=10)
        response.raise_for_status()
        await update.message.reply_text("✅ Conversation history cleared!")
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Error clearing history.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        user_question = update.message.text
        
        if user_question.startswith("/"):
            return
        
        await update.message.reply_text("🔍 Searching cookbook...")
        
        try:
            session_id = str(update.effective_user.id)
            
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": user_question,
                    "session_id": session_id,
                    "use_rerank": True,
                    "retrieval_k": 5,
                    "use_cache": True
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "Sorry, no answer found.")
            timing = data.get("timing", {})
            
            message = f"🍳 *Answer:*\n\n{answer}\n\n"
            
            message += f"\n⏱️ {timing.get('total', 0):.1f}s"
            if timing.get("cache_hit"):
                message += " (cached)"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except requests.exceptions.ConnectionError:
            await update.message.reply_text(
                "❌ *Error:* Cannot connect to the RAG server."
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            await update.message.reply_text(f"❌ *Error:* {str(e)}")


def run_bot():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        return
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Telegram bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()

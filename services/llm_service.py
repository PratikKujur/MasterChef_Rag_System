import os
from typing import Optional, Union
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from collections import deque
from core.config import groq_api_key, LLM_MODEL, GROQ_MODEL, MAX_HISTORY_SUMMARY_CHARS




def init_llm(model_name: str = LLM_MODEL) -> Union[ChatOllama, ChatGroq]:
    if groq_api_key:
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            api_key=groq_api_key,
            max_tokens=100,
        )
    return ChatOllama(
        model=model_name,
        temperature=0.2,
        base_url="http://localhost:11434",
        top_k=5,
        num_predict=100,
        top_p=0.9,
        repeat_penalty=1.1,
        keep_alive=30,
        num_ctx=2048,
    )

def create_rag_chain(llm: Union[ChatOllama, ChatGroq]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful cooking assistant. Use the following context from a cookbook to answer the user's question.
If you cannot find the answer in the context, say that you don't have that information.
IMPORTANT: Always cite your sources by including the page number in brackets at the end of relevant statements if context is available.
Don't generate sources separately, always include them in the answer if you use context information. If no context is relevant, just answer the question without citing sources.
Egample: "Add salt to taste [Page 5]", 

{history}

Context:
{context}"""),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


def create_summarize_chain(llm: Union[ChatOllama, ChatGroq]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You summarize conversation history briefly. Focus on key topics discussed. Keep under 200 characters."),
        ("human", "Summarize this conversation:\n{history}"),
    ])
    return prompt | llm | StrOutputParser()


def format_history(history: deque) -> str:
    if not history:
        return "No previous conversation history."
    
    formatted = ["Previous conversation:\n"]
    for i, (q, a) in enumerate(history, 1):
        formatted.append(f"Turn {i}:")
        formatted.append(f"  User: {q}")
        formatted.append(f"  Assistant: {a}\n")
    return "\n".join(formatted)


def summarize_history(history: deque, llm: Optional[Union[ChatOllama, ChatGroq]] = None) -> str:
    if not history:
        return "No previous conversation history."
    
    raw_history = format_history(history)
    
    if llm is None:
        if len(raw_history) > MAX_HISTORY_SUMMARY_CHARS:
            return raw_history[:MAX_HISTORY_SUMMARY_CHARS] + "..."
        return raw_history
    
    chain = create_summarize_chain(llm)
    summary = chain.invoke({"history": raw_history})
    return f"Conversation summary: {summary}"


def invoke_llm(chain, context: str, question: str, history: str) -> str:
    return chain.invoke({"context": context, "question": question, "history": history})

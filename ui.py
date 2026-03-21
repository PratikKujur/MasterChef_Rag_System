import gradio as gr
import requests
import uuid

API_URL = "http://localhost:8000"

def chat(message, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": message,
                "session_id": session_id,
                "use_rerank": True,
                "retrieval_k": 5,
                "use_cache": True
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return data["answer"], session_id
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to API. Make sure the server is running.", session_id
    except Exception as e:
        return f"Error: {str(e)}", session_id

def get_history(session_id):
    if not session_id:
        return "No session"
    try:
        response = requests.get(f"{API_URL}/history/{session_id}")
        data = response.json()
        if not data["history"]:
            return "No history"
        
        lines = [f"**Turn {i+1}:**\n**Q:** {h['question']}\n**A:** {h['answer']}" 
                 for i, h in enumerate(data["history"])]
        return "\n\n".join(lines)
    except:
        return "Error fetching history"

def clear_chat():
    return [], None

with gr.Blocks(title="MasterChef RAG Chat") as demo:
    gr.Markdown("# MasterChef RAG Chat")
    gr.Markdown("Chat with your cookbook using AI-powered retrieval.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Ask about recipes, cooking tips...", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            session_display = gr.Textbox(label="Session ID", interactive=False)
            history_btn = gr.Button("View History")
            history_display = gr.Markdown("No history")
    
    gr.Examples(
        examples=[
            "What is a simple recipe for chicken?",
            "How do I make pasta from scratch?",
            "What spices go well with fish?",
        ],
        inputs=msg,
    )
    
    def respond(message, history, session_id):
        if not message.strip():
            return "", history, session_id
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        bot_msg, session_id = chat(message, session_id)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_msg})
        return "", history, session_id
    
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, session_display],
        outputs=[msg, chatbot, session_display]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, session_display],
        outputs=[msg, chatbot, session_display]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, session_display]
    )
    
    history_btn.click(
        fn=get_history,
        inputs=[session_display],
        outputs=[history_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

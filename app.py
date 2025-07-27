# app_gradio.py

from parser import parse_query
from embedder import embed_text
from ranker import rank_results
import gradio as gr

def insurabot_response(user_query):
    parsed = parse_query(user_query)
    embedded = embed_text(parsed)
    results = rank_results(embedded)
    
    top_results = "\n\n".join(
        f"🔹 **{i+1}.** {res['title']}\n{res['snippet']}" for i, res in enumerate(results[:3])
    )
    
    return top_results or "Sorry, I couldn't find relevant answers."

demo = gr.Interface(
    fn=insurabot_response,
    inputs=gr.Textbox(lines=3, label="Ask InsuraBot a question"),
    outputs=gr.Markdown(label="Top Insurance Results"),
    title="🛡️ InsuraBot - AI Insurance Query Assistant",
    description="Type any insurance-related question and get AI-curated results!"
)

if __name__ == "__main__":
    demo.launch()

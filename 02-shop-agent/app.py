import gradio as gr
from agent import agent


def to_agent_history(history: list) -> list:
    """Convert Gradio's chat history into plain {role, content} messages."""
    cleaned = []
    for turn in history:
        content = turn.get("content")
        if isinstance(content, list):                    # some versions wrap text in parts
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if turn.get("role") in ("user", "assistant") and isinstance(content, str):
            cleaned.append({"role": turn["role"], "content": content})
    return cleaned


def chat(message: str, history: list) -> str:
    return agent(message, history=to_agent_history(history))


demo = gr.ChatInterface(
    fn=chat,
    title="🛍️ Smart Shop Assistant",
    description="Ask about prices or stock — watch the terminal to see the agent use its tools.",
    examples=["How much are the shoes?", "Is the hat in stock?", "Hi! What can you do?"],
)

if __name__ == "__main__":
    demo.launch()   # share=True only when recording a demo
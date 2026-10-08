import gradio as gr
from summarizer import summarize, PERSONALITIES

demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Website URL", placeholder="https://example.com"),
        gr.Dropdown(
            choices=list(PERSONALITIES),
            value="Friendly",
            label="Personality",
        ),
    ],
    outputs=gr.Markdown(label="Summary"),
    title="🔎 AI Website Summarizer",
    description="Paste a URL, pick a style, get a summary.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()   # add share=True for a temporary public link
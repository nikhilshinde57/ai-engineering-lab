# AI Engineering Lab

A portfolio of small, self-contained AI engineering projects. Each numbered folder is a
standalone app with its own README, dependencies, and `.env.example`. Folder names say
which pattern each project demonstrates.

| # | Project | Description |
|---|---------|-------------|
| 01 | [llm-pipeline-website-summarizer](01-llm-pipeline-website-summarizer/) | Scrapes a URL and summarizes it with a swappable LLM provider (Gemini/Groq/OpenAI), via a Gradio UI. |
| 02 | [agent-shop-assistant](02-agent-shop-assistant/) | A tool-calling agent that decides when to look up prices/stock, in a Gradio chat UI with memory. |
| 03 | [rag-pdf-chat](03-rag-pdf-chat/) | Chat with an uploaded PDF using retrieval-augmented generation, with grounded answers and page-level citations. |

## Setup

All projects share one virtualenv and one `.env` at the repo root. Each project's
`llm_client.py` finds the root `.env` by searching upward from its own folder.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r <project>/requirements.txt

cp 02-agent-shop-assistant/.env.example .env
# edit .env: set LLM_PROVIDER and the API key for that provider
```

Then `cd` into a project and follow its README.

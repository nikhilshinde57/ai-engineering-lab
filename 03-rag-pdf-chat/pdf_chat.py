import gradio as gr
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client import get_embeddings, get_langchain_model

MAX_PAGES = 60          # keep embedding calls within free-tier limits
TOP_K = 4

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
embeddings = get_embeddings()

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.

Rules:
- If the context doesn't contain the answer, reply exactly:
  "I couldn't find that in the document." and do NOT list any sources.
- Otherwise, after your answer, cite ONLY the pages whose text you actually used,
  as: Sources: page X, page Y.

Context:
{context}

Question: {question}
""")
chain = PROMPT | get_langchain_model(temperature=0) | StrOutputParser()


# ---------- Indexing (once per upload) ----------
def load_pdf(path: str) -> list[Document]:
    """One Document per page, tagged with its page number."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages[:MAX_PAGES]):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Document(page_content=text, metadata={"page": i + 1}))
    return pages


def build_index(path: str):
    pages = load_pdf(path)
    if not pages:
        raise ValueError("no extractable text (is it a scanned PDF?)")
    chunks = splitter.split_documents(pages)      # each chunk keeps its page metadata
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)                   # embed + store
    return store, len(pages), len(chunks)


# ---------- Querying (every question) ----------
def ask(store, question: str) -> str:
    results = store.similarity_search(question, k=TOP_K)
    print(f"\n❓ {question}")
    for d in results:
        print(f"   🔎 page {d.metadata['page']}: {d.page_content[:80]!r}...")
    context = "\n\n".join(f"[page {d.metadata['page']}] {d.page_content}" for d in results)
    return chain.invoke({"context": context, "question": question})


# ---------- UI ----------
state = {"store": None}     # survives across Gradio events, so we index only once


def on_upload(path):
    if not path:
        return "Upload a PDF to start."
    try:
        store, n_pages, n_chunks = build_index(path)
    except Exception as e:
        return f"⚠️ Could not index the PDF: {e}"
    state["store"] = store
    return f"✅ Indexed {n_pages} pages into {n_chunks} chunks. Ask away!"


def chat(message, history):
    if state["store"] is None:
        return "Please upload a PDF first 📄"
    try:
        return ask(state["store"], message)
    except Exception as e:
        return f"⚠️ {type(e).__name__}: the model may be rate-limited, try again in ~30s."


with gr.Blocks(title="Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF (RAG)")
    pdf = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
    status = gr.Markdown("Upload a PDF to start.")
    pdf.upload(on_upload, inputs=pdf, outputs=status)
    gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch()
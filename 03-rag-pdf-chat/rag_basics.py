from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_client import get_embeddings, get_langchain_model

# ---------- Phase 1: INDEXING (once) ----------
docs = [
    Document(page_content="Our return policy allows refunds within 30 days of purchase.", metadata={"source": "returns.md"}),
    Document(page_content="Shipping is free for orders above ₹999 across India.", metadata={"source": "shipping.md"}),
    Document(page_content="Damaged or defective items can be exchanged at no cost; email support with a photo.", metadata={"source": "defects.md"}),
    Document(page_content="For corporate orders above 50 units, contact sales@example.com.", metadata={"source": "sales.md"}),
    Document(page_content="Our office is in Baner, Pune. Open Mon-Fri 10am-7pm.", metadata={"source": "contact.md"}),
]

store = InMemoryVectorStore(get_embeddings())
store.add_documents(docs)                     # embeds each doc and stores the vectors
print(f"Indexed {len(docs)} documents\n")

# ---------- Phase 2: QUERYING (every question) ----------
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know."
Mention which source you used.

Context:
{context}

Question: {question}
""")
chain = prompt | get_langchain_model(temperature=0) | StrOutputParser()


def rag_answer(question: str, k: int = 2) -> str:
    results = store.similarity_search_with_score(question, k=k)   # the librarian
    print(f"❓ {question}")
    for doc, score in results:
        print(f"   🔎 {score:.2f}  [{doc.metadata['source']}]  {doc.page_content}")

    context = "\n".join(f"[{d.metadata['source']}] {d.page_content}" for d, _ in results)
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    for q in [
        "How long do I have to send something back?",   # no shared words with the returns doc
        "My product arrived broken, what do I do?",     # 'broken' never appears in the docs
        "Who is the CEO of the company?",               # not in the docs at all
    ]:
        print("💬", rag_answer(q), "\n")
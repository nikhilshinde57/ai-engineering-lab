from numpy import dot
from numpy.linalg import norm
from llm_client import get_embeddings

emb = get_embeddings()


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


base = emb.embed_query("A cat is sleeping on the couch.")
print("Vector length:", len(base))
print("First 5 numbers:", [round(x, 3) for x in base[:5]])

tests = [
    "A kitten is napping on the sofa.",      # same meaning, different words
    "The dog is running in the park.",       # related topic
    "The RBI cut the repo rate by 25 bps.",  # unrelated
]
for t in tests:
    print(f"{cosine(base, emb.embed_query(t)):.2f}  ←  {t}")
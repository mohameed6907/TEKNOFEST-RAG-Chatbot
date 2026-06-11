import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.retrievers import build_local_docs_retriever, retrieve_from_vectorstore
from app.rag.text_cleaner import normalize_turkish_query

settings = get_settings()
vs = build_local_docs_retriever(settings)
query = normalize_turkish_query("Blokzincir Yarışması ne")

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

print("Normalized query:", query)

chunks = retrieve_from_vectorstore(vs, query=query, source_type="local_docs", k=10)
for i, c in enumerate(chunks):
    preview = c.content[:150].replace('\n', ' ')
    print(f"[{i}] Source: {c.source} | Score: {c.score:.4f} | Preview: {preview}")

import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.retrievers import build_local_docs_retriever

settings = get_settings()
vs = build_local_docs_retriever(settings)

# Let's inspect all chunks in local_docs
col = vs._collection
results = col.get(include=["metadatas"])
sources = set()
for m in results.get("metadatas", []):
    if m and "source" in m:
        sources.add(m["source"])

print("Indexed sources in local_docs:")
for s in sorted(sources):
    print("-", s)

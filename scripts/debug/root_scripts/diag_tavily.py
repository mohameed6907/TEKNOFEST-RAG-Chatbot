import sys
sys.path.insert(0, ".")
from app.config import get_settings
settings = get_settings()
print("TAVILY_API_KEY set:", bool(settings.tavily_api_key))

import chromadb
client = chromadb.PersistentClient(path="RAG/chroma_local_docs")
cols = client.list_collections()
col = cols[0] if cols else None
if col:
    from app.rag.embedding_service import EmbeddingService
    svc = EmbeddingService(settings)
    emb = svc.embed_query("robolig yarışması ödül miktarı")
    r = col.query(query_embeddings=[emb], n_results=3, include=["distances","documents"])
    print("Robolig ödül query distances:", r["distances"][0])
    print("Confidence values:", [round(1-(d/2),3) for d in r["distances"][0]])

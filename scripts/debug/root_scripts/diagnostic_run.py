import sys, os
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
import chromadb

settings = get_settings()
svc = EmbeddingService(settings)

queries = [
    "sağlıkta yapay zeka TEKNOFEST yarışması",
    "TEKNOFEST yapay zeka kategorisi",
    "Sağlık ve biyoteknoloji yarışması"
]

client = chromadb.PersistentClient(path="RAG/chroma_local_docs")
cols = client.list_collections()
if not cols:
    print("HATA: Hiç collection yok!")
else:
    col = cols[0]
    print(f"Collection: {col.name}, Chunk sayısı: {col.count()}, Metric: {col.metadata}")
    for q in queries:
        emb = svc.embed_query(q)
        results = col.query(query_embeddings=[emb], n_results=5, include=["documents","metadatas","distances"])
        print(f"\nQuery: '{q}'")
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            confidence = 1 - (dist / 2)
            src = meta.get("source", "?")
            print(f"  dist={dist:.4f} | conf={confidence:.4f} | {src} | {doc[:80]}...")

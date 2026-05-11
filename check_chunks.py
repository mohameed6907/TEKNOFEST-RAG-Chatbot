# check_chunks.py
import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
import chromadb

settings = get_settings()
client = chromadb.PersistentClient(path=str(settings.rag_root / "chroma_local_docs"))
collection = client.get_collection(settings.chroma_local_collection)

embed_svc = EmbeddingService(settings)
query_embedding = embed_svc.embed_query("robolig alt kategori yarışma")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("=== CHUNK İÇERİKLERİ ===")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Kaynak : {meta.get('source', '?')}")
    print(f"Dist   : {dist:.3f}")
    print(f"İçerik :\n{doc[:500]}")
    print("-" * 60)
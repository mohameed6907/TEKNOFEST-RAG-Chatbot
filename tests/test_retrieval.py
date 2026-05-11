"""Phase 1 — Real retrieval pipeline testi"""
import sys
sys.path.insert(0, ".")

from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
import chromadb

settings = get_settings()
svc = EmbeddingService(settings)

test_query = "TEKNOFEST yarışma kategorileri"
embedding = svc.embed_query(test_query)
print(f"Query embedding dim: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Chroma'dan çek
client = chromadb.PersistentClient(path="RAG/chroma_local_docs")
collections = client.list_collections()
print(f"\nCollections: {[c.name for c in collections]}")
if collections:
    col = collections[0]
    print(f"Collection metadata: {col.metadata}")
    results = col.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\nTop 5 results for '{test_query}':")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1.0 - (dist / 2.0)
        print(f"\n[{i+1}] Distance: {dist:.4f} → Similarity: {similarity:.4f}")
        print(f"  Source: {meta.get('source', 'unknown')}")
        print(f"  Content[:250]: {doc[:250]}")
else:
    print("No collections found!")

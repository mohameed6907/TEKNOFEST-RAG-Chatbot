"""Retrieval test - ASCII safe output"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
import chromadb

settings = get_settings()
svc = EmbeddingService(settings)
test_query = 'TEKNOFEST yarisma kategorileri'
embedding = svc.embed_query(test_query)
print(f'Query embedding dim: {len(embedding)}')

client = chromadb.PersistentClient(path='RAG/chroma_local_docs')
collections = client.list_collections()
col = collections[0]
print(f'Collection metadata: {col.metadata}')
keys = ["documents", "metadatas", "distances"]
results = col.query(query_embeddings=[embedding], n_results=5, include=keys)
docs = results["documents"][0]
metas = results["metadatas"][0]
dists = results["distances"][0]
for i in range(len(docs)):
    dist = dists[i]
    sim = 1.0 - (dist / 2.0)
    src = metas[i].get("source", "?")[-50:]
    print(f'[{i+1}] dist={dist:.4f} sim={sim:.4f} src=...{src}')
    print(f'  Content: {docs[i][:200]}')
    print()

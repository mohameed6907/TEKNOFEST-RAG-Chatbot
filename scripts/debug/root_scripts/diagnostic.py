import sys, os, json
sys.path.insert(0, ".")

print("=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Collection health
try:
    import chromadb
    for name, path in [("local_docs", "RAG/chroma_local_docs"), ("site", "RAG/chroma_teknofest_site")]:
        if os.path.exists(path):
            client = chromadb.PersistentClient(path=path)
            cols = client.list_collections()
            total = sum(c.count() for c in cols)
            col_names = [c.name for c in cols]
            print(f"\nChroma '{name}': {total} chunks, collections: {col_names}")
            if cols and total > 0:
                sample = cols[0].get(limit=1, include=["documents","metadatas","embeddings"])
                emb = sample["embeddings"][0] if sample["embeddings"] else None
                print(f"  Embedding dim: {len(emb) if emb else 'MISSING'}")
                print(f"  Sample metadata: {sample['metadatas'][0]}")
                print(f"  Sample content: {sample['documents'][0][:100]}...")
                col_meta = cols[0].metadata or {}
                print(f"  Collection metadata (metric): {col_meta}")
        else:
            print(f"\nChroma '{name}': PATH NOT FOUND -- {path}")
except Exception as e:
    print(f"\nChroma ERROR: {e}")

# 2. Test retrieval with a real query
try:
    from app.config import get_settings
    from app.rag.embedding_service import EmbeddingService
    settings = get_settings()
    svc = EmbeddingService(settings)
    emb = svc.embed_query("TEKNOFEST yarışma kategorileri")
    print(f"\nQuery embedding: dim={len(emb)}, first3={emb[:3]}")

    client = chromadb.PersistentClient(path="RAG/chroma_local_docs")
    cols = client.list_collections()
    if cols:
        results = cols[0].query(query_embeddings=[emb], n_results=3, include=["documents","metadatas","distances"])
        print(f"\nRetrieval test -- top 3 distances: {results['distances'][0]}")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"  [{i+1}] dist={dist:.4f} | {doc[:80]}...")
except Exception as e:
    print(f"\nRetrieval test ERROR: {e}")

# 3. Read last 5 eval log entries
try:
    log_path = "RAG/eval_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        last5 = [json.loads(l) for l in lines[-5:]]
        print(f"\nLast 5 eval log entries:")
        for e in last5:
            print(f"  route={e.get('route')} | hallucination={e.get('hallucination_status')} | chunks_retrieved={e.get('retrieved_count')} | chunks_used={e.get('selected_count')} | query={e.get('query','')[:50]}")
    else:
        print(f"\nEval log: NOT FOUND")
except Exception as e:
    print(f"\nEval log ERROR: {e}")

print("\n" + "=" * 60)

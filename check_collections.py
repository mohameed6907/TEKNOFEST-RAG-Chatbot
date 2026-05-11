import chromadb
# local_docs
c1 = chromadb.PersistentClient(path="RAG/chroma_local_docs")
for col in c1.list_collections():
    m = col.metadata or {}
    print(f"[local_docs] collection='{col.name}' count={col.count()} metric={m.get('hnsw:space','NOT SET')} emb_model={m.get('embedding_model','?')}")

# site
c2 = chromadb.PersistentClient(path="RAG/chroma_teknofest_site")
cols2 = c2.list_collections()
if not cols2:
    print("[teknofest_site] NO COLLECTIONS FOUND")
else:
    for col in cols2:
        m = col.metadata or {}
        print(f"[teknofest_site] collection='{col.name}' count={col.count()} metric={m.get('hnsw:space','NOT SET')} emb_model={m.get('embedding_model','?')}")

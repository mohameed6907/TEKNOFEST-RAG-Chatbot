"""Phase 1 — ChromaDB durumu teşhis scripti"""
import chromadb
import os

for collection_name in ["chroma_local_docs", "chroma_teknofest_site"]:
    path = f"RAG/{collection_name}"
    if os.path.exists(path):
        client = chromadb.PersistentClient(path=path)
        collections = client.list_collections()
        print(f"\n=== {collection_name} ===")
        print(f"Collections found: {[c.name for c in collections]}")
        for col in collections:
            count = col.count()
            print(f"  Collection name : {col.name}")
            print(f"  Chunk count     : {count}")
            try:
                print(f"  Coll metadata   : {col.metadata}")
            except Exception as e:
                print(f"  Coll metadata   : ERROR {e}")
            if count > 0:
                sample = col.get(limit=3, include=["documents", "metadatas", "embeddings"])
                for i, doc in enumerate(sample["documents"]):
                    print(f"  --- Chunk [{i}] ---")
                    print(f"    Content[:200]: {doc[:200]!r}")
                    print(f"    Metadata     : {sample['metadatas'][i]}")
                    emb_list = sample["embeddings"]
                    emb = emb_list[i] if emb_list is not None and len(emb_list) > i else None
                    if emb is not None:
                        import numpy as np
                        emb_arr = list(emb) if hasattr(emb, '__iter__') else []
                        print(f"    Embed dim    : {len(emb_arr)}, first3: {emb_arr[:3]}")
                    else:
                        print("    Embeddings   : MISSING!")
    else:
        print(f"\n=== {collection_name} — PATH NOT FOUND ===")

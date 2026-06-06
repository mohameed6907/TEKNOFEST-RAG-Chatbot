"""
recreate_cosine_collections.py
================================
Mevcut Chroma collection'larını silerek cosine distance metric ile yeniden oluşturur.
Veriler korunmaz — ardından ingest_local_docs.py ve crawl_teknofest.py çalıştırılmalıdır.

Run:
    python recreate_cosine_collections.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb

print("=== Chroma Collection Cosine Fix ===\n")

for col_name, db_path in [
    ("local_docs", "RAG/chroma_local_docs"),
    ("teknofest_site", "RAG/chroma_teknofest_site"),
]:
    path = Path(db_path)
    if not path.exists():
        print(f"[{col_name}] Path does not exist, creating: {db_path}")
        path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))
    cols = client.list_collections()
    existing_names = [c.name for c in cols]
    print(f"[{col_name}] Existing collections: {existing_names}")

    if col_name in existing_names:
        old_col = client.get_collection(col_name)
        old_count = old_col.count()
        old_meta = old_col.metadata
        print(f"[{col_name}] Current chunk count: {old_count}, metadata: {old_meta}")
        print(f"[{col_name}] Deleting collection '{col_name}'...")
        client.delete_collection(col_name)
        print(f"[{col_name}] Deleted.")

    # Cosine metric ile yeni collection oluştur
    new_col = client.create_collection(
        name=col_name,
        metadata={
            "hnsw:space": "cosine",
            "description": f"TEKNOFEST RAG - {col_name}",
        }
    )
    print(f"[{col_name}] Created with cosine metric. Chunk count: {new_col.count()}")
    print(f"[{col_name}] Collection metadata: {new_col.metadata}")
    print()

print("=== Done! ===")
print("Simdi su komutlari calistirin:")
print("  1. python scripts/ingest_local_docs.py")
print("  2. python scripts/crawl_teknofest.py")

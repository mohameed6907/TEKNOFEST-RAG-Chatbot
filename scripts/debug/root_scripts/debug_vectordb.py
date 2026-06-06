# debug_vectordb.py — Proje kök dizinine kaydet ve çalıştır

import sys
sys.path.insert(0, ".")  # Proje kökünden çalıştır

from app.config import get_settings
from app.rag.embedding_service import EmbeddingService
# Bunu debug_vectordb.py'nin en üstüne ekle, geçici olarak
import chromadb
from app.config import get_settings
settings = get_settings()

client = chromadb.PersistentClient(path=str(settings.rag_root / "chroma_local_docs"))
print("Mevcut collection'lar:", client.list_collections())

# ─── 1. Chroma'ya bağlan ───────────────────────────────────────
client = chromadb.PersistentClient(path=str(settings.rag_root / "chroma_local_docs"))
collection = client.get_collection("chroma_local_docs")

# ─── 2. Kaç chunk var? ────────────────────────────────────────
total = collection.count()
print(f"\n[1] Toplam chunk sayısı: {total}")
if total == 0:
    print("    ❌ SORUN: Vector DB boş! PDF hiç index'lenmemiş.")
    sys.exit()

# ─── 3. Kaynak listesi ─────────────────────────────────────────
all_meta = collection.get(include=["metadatas"])["metadatas"]
sources = set(m.get("source", "unknown") for m in all_meta)
print(f"\n[2] Index'lenmiş kaynaklar ({len(sources)} dosya):")
for s in sorted(sources):
    count = sum(1 for m in all_meta if m.get("source") == s)
    print(f"    • {s}  ({count} chunk)")

robolig_found = any("obolig" in s or "Robolig" in s or "robolig" in s for s in sources)
if not robolig_found:
    print("\n    ❌ SORUN: Robolig şartnamesi hiç index'lenmemiş!")
    print("    → /api/admin/ingest endpoint'ini çalıştır veya ingest script'ini elle başlat.")
    sys.exit()
else:
    print("\n    ✅ Robolig dosyası index'te mevcut.")

# ─── 4. Robolig chunk'larından örnek ──────────────────────────
robolig_chunks = [
    (m, i) for i, m in enumerate(all_meta)
    if "obolig" in m.get("source", "").lower() or "robolig" in m.get("source", "").lower()
]
print(f"\n[3] Robolig chunk sayısı: {len(robolig_chunks)}")
sample = collection.get(
    ids=[collection.get()["ids"][i] for _, i in robolig_chunks[:3]],
    include=["documents", "metadatas"]
)
for doc, meta in zip(sample["documents"], sample["metadatas"]):
    print(f"\n    Chunk başı: {doc[:150]}...")
    print(f"    Metadata: {meta}")

# ─── 5. Embedding araması testi ───────────────────────────────
embed_svc = EmbeddingService(settings)
test_queries = [
    "robolig yarışması kaç kişilik takım",
    "TEKNOFEST Robolig 2026 şartname",
    "robotik yarışma takım üye sayısı",
]

print("\n[4] Embedding arama testleri:")
for query in test_queries:
    query_embedding = embed_svc.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\n    Sorgu: '{query}'")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = round(1 - dist / 2, 3)  # L2→similarity
        source = meta.get("source", "?")
        is_robolig = "obolig" in source.lower()
        tag = "✅ ROBOLIG" if is_robolig else "  ·"
        print(f"    {tag}  sim={similarity:.3f}  dist={dist:.3f}  src={source[:50]}")
        if is_robolig:
            print(f"           içerik: {doc[:100]}...")
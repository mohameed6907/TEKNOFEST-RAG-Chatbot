#!/usr/bin/env python
"""
ChromaDB local_docs collection'ından TEKNOFEST_Ansiklopedi.pdf chunk'larını sil.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import chromadb
from app.config import get_settings

settings = get_settings()

print("=" * 60)
print("TEKNOFEST_Ansiklopedi.pdf chunk'larını silme")
print("=" * 60)

client = chromadb.PersistentClient(path=str(settings.chroma_local_docs_path))
collection = client.get_collection(settings.chroma_local_collection)

before_count = collection.count()
print(f"Silme öncesi toplam chunk: {before_count}")

# Tüm metadata'ları çek ve Ansiklopedi olanları bul
all_data = collection.get(include=["metadatas"])
ansiklopedi_ids = []
for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
    source = (meta or {}).get("source", "")
    if "Ansiklopedi" in source or "ansiklopedi" in source.lower():
        ansiklopedi_ids.append(doc_id)

print(f"Silinecek Ansiklopedi chunk sayısı: {len(ansiklopedi_ids)}")

if ansiklopedi_ids:
    # Batch halinde sil
    batch_size = 100
    for i in range(0, len(ansiklopedi_ids), batch_size):
        batch = ansiklopedi_ids[i:i+batch_size]
        collection.delete(ids=batch)
        print(f"  Silindi: {min(i+batch_size, len(ansiklopedi_ids))}/{len(ansiklopedi_ids)}")

    after_count = collection.count()
    print(f"\n✅ Silme tamamlandı!")
    print(f"Silme öncesi: {before_count} chunk")
    print(f"Silme sonrası: {after_count} chunk")
    print(f"Silinen: {before_count - after_count} chunk")
    
    # Doğrulama: Ansiklopedi hâlâ var mı?
    verify_data = collection.get(include=["metadatas"])
    remaining = [
        doc_id for doc_id, meta in zip(verify_data["ids"], verify_data["metadatas"])
        if "ansiklopedi" in ((meta or {}).get("source", "")).lower()
    ]
    print(f"\n🔍 Doğrulama: Kalan Ansiklopedi chunk'ları: {len(remaining)}")
    if len(remaining) == 0:
        print("✅ Tüm Ansiklopedi chunk'ları başarıyla silindi!")
    else:
        print(f"⚠️  Hâlâ {len(remaining)} Ansiklopedi chunk'ı kaldı!")
    
    # Kalan source'ları listele
    sources = set()
    for meta in verify_data["metadatas"]:
        sources.add((meta or {}).get("source", ""))
    print(f"\nKalan source'lar:")
    for s in sorted(sources):
        count = sum(1 for m in verify_data["metadatas"] if (m or {}).get("source", "") == s)
        print(f"  [{count:3d}] {s}")
else:
    print("Ansiklopedi chunk'ı bulunamadı, temiz.")

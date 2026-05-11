#!/usr/bin/env python
"""
ChromaDB'deki TÜM unique source değerlerini listele ve 
Ansiklopedi ile ilgili chunk'ları daha geniş arama ile bul.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import chromadb
from app.config import get_settings

settings = get_settings()

collections_to_check = [
    ("local_docs", str(settings.chroma_local_docs_path), settings.chroma_local_collection),
    ("teknofest_site", str(settings.chroma_teknofest_site_path), settings.chroma_site_collection),
]

for label, path, col_name in collections_to_check:
    print(f"\n{'='*60}")
    print(f"{label} ({col_name})")
    print(f"{'='*60}")
    
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(col_name)
        total = collection.count()
        print(f"Toplam chunk: {total}")
        
        # Tüm metadata'ları çek
        all_data = collection.get(include=["metadatas", "documents"])
        
        # Unique source'ları bul
        sources = set()
        ansiklopedi_ids = []
        ansiklopedi_by_content = []
        
        for doc_id, meta, doc_text in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
            meta = meta or {}
            source = meta.get("source", "")
            sources.add(source)
            
            # Source'ta Ansiklopedi ara
            if "ansiklopedi" in source.lower() or "Ansiklopedi" in source:
                ansiklopedi_ids.append(doc_id)
            
            # İçerikte Ansiklopedi ara (belki metadata'da değil ama content'te var)
            if doc_text and ("ansiklopedi" in doc_text.lower()):
                ansiklopedi_by_content.append((doc_id, source, doc_text[:100]))
        
        print(f"\nUnique source'lar ({len(sources)} adet):")
        for s in sorted(sources):
            # Her source'dan kaç chunk var
            count = sum(1 for m in all_data["metadatas"] if (m or {}).get("source", "") == s)
            print(f"  [{count:3d}] {s}")
        
        if ansiklopedi_ids:
            print(f"\n⚠️  Source'ta Ansiklopedi bulunan chunk'lar: {len(ansiklopedi_ids)}")
        
        if ansiklopedi_by_content:
            print(f"\n⚠️  İçerikte 'ansiklopedi' geçen chunk'lar: {len(ansiklopedi_by_content)}")
            for doc_id, source, preview in ansiklopedi_by_content[:5]:
                print(f"    ID: {doc_id} | Source: {source}")
                print(f"    Content: {preview}...")
                
    except Exception as e:
        print(f"  HATA: {e}")
        import traceback
        traceback.print_exc()

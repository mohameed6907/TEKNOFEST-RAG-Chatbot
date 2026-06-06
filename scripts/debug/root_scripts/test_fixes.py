#!/usr/bin/env python
"""
Test: İnsansız Kara Araçları Yarışması ödülleri sorgusu.
Beklenen:
  1. Kaynaklarda Ansiklopedi geçmemeli
  2. Cevap sonunda dış site yönlendirmesi olmamalı
"""
import sys
import io
import json
import asyncio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

async def main():
    settings = get_settings()
    graph = build_teknofest_graph(settings)
    
    question = "İnsansız Kara Araçları Yarışması ödülleri nedir?"
    print(f"SORU: {question}")
    print("=" * 70)
    
    result = await run_graph(graph, question=question)
    
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    route = result.get("route_taken", "")
    meta = result.get("meta", {})
    
    print(f"\nROTE: {route}")
    print(f"\nCEVAP:\n{answer}")
    
    print(f"\nKAYNAKLAR ({len(sources)} adet):")
    for i, src in enumerate(sources):
        src_path = src.get("metadata", {}).get("source", "unknown")
        src_type = src.get("type", "unknown")
        score = src.get("score", "N/A")
        print(f"  [{i+1}] type={src_type} | score={score}")
        print(f"       source={src_path}")
    
    # Doğrulama kontrolleri
    print("\n" + "=" * 70)
    print("DOĞRULAMA:")
    
    # Kontrol 1: Ansiklopedi kaynaklarda olmamalı
    ansiklopedi_found = False
    for src in sources:
        source_path = str(src.get("metadata", {}).get("source", ""))
        if "ansiklopedi" in source_path.lower() or "Ansiklopedi" in source_path:
            ansiklopedi_found = True
            print(f"  ❌ Ansiklopedi hâlâ kaynaklarda: {source_path}")
    if not ansiklopedi_found:
        print("  ✅ Kaynaklarda Ansiklopedi YOK — BAŞARILI")
    
    # Kontrol 2: Dış site yönlendirmesi olmamalı
    redirect_phrases = [
        "ziyaret edebilir", "ziyaret edin",
        "resmi site", "resmi web",
        "teknofest.org adres",
        "kontrol edebilir", "kontrol edin",
        "bakmanı öneririm", "bakmanızı",
    ]
    redirect_found = False
    for phrase in redirect_phrases:
        if phrase in answer.lower():
            redirect_found = True
            print(f"  ❌ Yasak ifade bulundu: '{phrase}'")
    if not redirect_found:
        print("  ✅ Cevap sonunda dış site yönlendirmesi YOK — BAŞARILI")
    
    # Meta bilgileri
    print(f"\nMETA:")
    for key in ["intent", "local_confidence", "site_confidence", "tavily_used", "reranker_used", "hallucination_check"]:
        if key in meta:
            print(f"  {key}: {meta[key]}")

if __name__ == "__main__":
    asyncio.run(main())

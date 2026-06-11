import asyncio
import json
import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    settings = get_settings()
    graph = build_teknofest_graph(settings)
    
    q = "Blokzincir Yarışması ne"
    res = await run_graph(graph, question=q, chat_history=[], metadata={"session_id": "test_session", "user_id": "test_user"})
    print("--- RESPONSE ---")
    print("Answer:", res.get("answer"))
    print("Route Taken:", res.get("route_taken"))
    print("Meta:")
    print(json.dumps(res.get("meta", {}), indent=2, ensure_ascii=False))
    print("\nContext Chunks count:", len(res.get("context_chunks", [])))
    for i, c in enumerate(res.get("context_chunks", [])):
        print(f"[{i}] Source: {c.source}")
        print(f"    Preview: {c.content[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())

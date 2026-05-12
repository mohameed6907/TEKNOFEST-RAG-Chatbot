import asyncio
import json
from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

settings = get_settings()
graph = build_teknofest_graph(settings)

async def test():
    history = []
    queries = [
        "İKA yarışması için daha fazla bilgi verir misin",
        "parkur hakkında daha fazla bilgi var mı",
        "yarışma parkuru hakkında daha fazla bilgi mevcut mu"
    ]
    
    for q in queries:
        print(f"\n--- QUERY: {q} ---")
        res = await run_graph(graph, q, chat_history=list(history))
        print("Route:", res.get("route_taken"))
        print("Meta:", json.dumps(res.get("meta", {}), ensure_ascii=False, indent=2))
        
        # update history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": res.get("answer", "")[:200]})

asyncio.run(test())

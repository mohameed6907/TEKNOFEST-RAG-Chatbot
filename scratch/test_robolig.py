import asyncio
import json
from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

settings = get_settings()
graph = build_teknofest_graph(settings)

async def test():
    q = "robolig yarışması hakkında bilgi verir misin"
    print(f"\n--- QUERY: {q} ---")
    res = await run_graph(graph, q)
    print("Route:", res.get("route_taken"))
    print("Meta:", json.dumps(res.get("meta", {}), ensure_ascii=False, indent=2))
    print("Answer:", res.get("answer"))

asyncio.run(test())

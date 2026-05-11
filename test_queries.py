import asyncio
import json
import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

async def main():
    settings = get_settings()
    graph = build_teknofest_graph(settings)
    
    q1 = "İnsansız Kara Araçları Yarışması 2026 ödülleri nedir?"
    res1 = await run_graph(graph, question=q1, chat_history=[], metadata={"session_id": "1", "user_id": "test"})
    print("--- Q1 RESPONSE ---")
    print(json.dumps(res1, indent=2, ensure_ascii=False, default=str))
    
    q2 = "yarışmanın geçen yılki birincileri kim olmuş"
    history = [{"role": "user", "content": q1}, {"role": "assistant", "content": res1.get("answer", "")}]
    res2 = await run_graph(graph, question=q2, chat_history=history, metadata={"session_id": "1", "user_id": "test"})
    print("--- Q2 RESPONSE ---")
    print(json.dumps(res2, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    asyncio.run(main())

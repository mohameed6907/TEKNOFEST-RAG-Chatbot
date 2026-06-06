import asyncio
import logging
import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.rag.graph import build_teknofest_graph, run_graph

logging.basicConfig(level=logging.DEBUG)

async def main():
    settings = get_settings()
    graph = build_teknofest_graph(settings)
    
    q1 = "İnsansız kara aracı yarışmasının ödülü kaç TL"
    print(f"Testing Q1: {q1}")
    res1 = await run_graph(graph, question=q1, chat_history=[], metadata={"session_id": "1", "user_id": "test"})
    print("Answer 1:", res1.get("answer"))
    
    q2 = "yarışmanın geçen yılki birincileri kim olmuş"
    print(f"Testing Q2: {q2}")
    history = [{"role": "user", "content": q1}, {"role": "assistant", "content": res1.get("answer", "")}]
    res2 = await run_graph(graph, question=q2, chat_history=history, metadata={"session_id": "1", "user_id": "test"})
    print("Answer 2:", res2.get("answer"))

if __name__ == "__main__":
    asyncio.run(main())

"""
Debug test to trace the flow
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.config import get_settings
from app.rag.graph import build_teknofest_graph, GraphState

async def test_query_debug():
    settings = get_settings()
    graph = build_teknofest_graph(settings=settings)
    
    question = "ben sağlıkta yapay zeka alanında yarışacağım"
    
    initial_state = {
        "question": question,
        "chat_history": [],
        "rephrased_question": "",
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    print(f"Intent: {final_state.get('intent')}")
    print(f"Route taken: {final_state.get('route_taken')}")
    print(f"Rephrased: {final_state.get('rephrased_question')}")
    print(f"Retrieved chunks count: {len(final_state.get('retrieved_chunks', []))}")
    print(f"Context chunks count: {len(final_state.get('context_chunks', []))}")
    print(f"Context string length: {len(final_state.get('context_str', ''))}")
    print(f"Context string preview: {final_state.get('context_str', '')[:200]}")
    print(f"Answer: {final_state.get('answer')}")
    print(f"Meta hallucination_check: {final_state.get('meta', {}).get('hallucination_check')}")
    
    if final_state.get('context_chunks'):
        print(f"\nContext chunks details:")
        for i, ch in enumerate(final_state.get('context_chunks', [])[:3]):
            print(f"  [{i}] {ch.source_type} | {ch.metadata.get('source', '?')} | {ch.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_query_debug())

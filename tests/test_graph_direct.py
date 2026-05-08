"""
Test script to validate the graph without needing the full server
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.config import get_settings
from app.rag.graph import build_teknofest_graph

async def test_queries():
    settings = get_settings()
    graph = build_teknofest_graph(settings=settings)
    
    test_cases = [
        ("ben sağlıkta yapay zeka alanında yarışacağım", None),
        ("bu kategoride nelere dikkat etmeliyim", [
            {"role": "user", "content": "ben sağlıkta yapay zeka alanında yarışacağım"},
            {"role": "assistant", "content": "Sağlıkta yapay zeka kategorisinde yarışmaya hazırlanıyorsunuz. Bu kategori oldukça kompetitif..."},
        ]),
        ("yarışma hakkında önerin var mı", [
            {"role": "user", "content": "ben sağlıkta yapay zeka alanında yarışacağım"},
            {"role": "assistant", "content": "Sağlıkta yapay zeka kategorisinde yarışmaya hazırlanıyorsunuz."},
            {"role": "user", "content": "bu kategoride nelere dikkat etmeliyim"},
            {"role": "assistant", "content": "Birkaç önemli noktaya dikkat etmelisiniz..."},
        ]),
    ]
    
    for i, (question, history) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question}")
        print(f"History: {len(history or [])} messages")
        print('='*60)
        
        try:
            from app.rag.graph import run_graph
            result = await run_graph(
                graph=graph,
                question=question,
                chat_history=history or []
            )
            
            print(f"Route: {result.get('route_taken', '?')}")
            print(f"Answer preview: {result.get('answer', '')[:200]}...")
            print(f"Sources count: {len(result.get('sources', []))}")
            
            if result.get('sources'):
                print("Source details:")
                for src in result['sources'][:2]:
                    print(f"  - {src.get('type')} | {src.get('metadata', {}).get('source', '?')}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_queries())

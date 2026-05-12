import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag.graph import build_teknofest_graph, run_graph
from app.config import get_settings

# Force UTF-8 output for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def test_rephrase():
    settings = get_settings()
    # Ensure temperature 0 for rephrase is handled by the node correctly
    graph = build_teknofest_graph(settings)
    
    session_id = "test_rephrase_session"
    
    # Query 1
    q1 = "İKA yarışması için daha fazla bilgi verir misin"
    print(f"Executing Query 1: {q1}")
    res1 = await run_graph(graph, q1, chat_history=[], metadata={"session_id": session_id})
    
    # Query 2
    # Truncate assistant response as chat.py would
    ans1_truncated = res1['answer'][:200] + "..." if len(res1['answer']) > 200 else res1['answer']
    history2 = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": ans1_truncated}
    ]
    q2 = "parkur hakkında daha fazla bilgi var mı"
    print(f"Executing Query 2: {q2}")
    res2 = await run_graph(graph, q2, chat_history=history2, metadata={"session_id": session_id})
    
    # Query 3
    ans2_truncated = res2['answer'][:200] + "..." if len(res2['answer']) > 200 else res2['answer']
    history3 = history2 + [
        {"role": "user", "content": q2},
        {"role": "assistant", "content": ans2_truncated}
    ]
    q3 = "yarışma parkuru hakkında daha fazla bilgi mevcut mu"
    print(f"Executing Query 3: {q3}")
    res3 = await run_graph(graph, q3, chat_history=history3, metadata={"session_id": session_id})
    
    # Read last 3 lines of eval log
    eval_log_path = settings.eval_log_path
    logs = []
    if eval_log_path.exists():
        with open(eval_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-3:]:
                logs.append(json.loads(line))
    
    print("\n--- EVAL LOG RESULTS ---")
    for log in logs:
        print(json.dumps({
            "original_query": log.get("query"),
            "rephrased_question": log.get("rephrased_question"),
            "route": log.get("route")
        }, indent=2, ensure_ascii=False))

    # Full JSON responses for the user
    print("\n--- FULL JSON RESPONSES (META ONLY) ---")
    print(json.dumps({
        "q1_meta": res1.get("meta"),
        "q2_meta": res2.get("meta"),
        "q3_meta": res3.get("meta")
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_rephrase())

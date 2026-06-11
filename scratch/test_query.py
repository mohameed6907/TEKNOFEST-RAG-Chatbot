import asyncio
import sys
import os
import logging

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from app.config import get_settings
from app.rag import graph as graph_mod
from app.rag.graph import build_teknofest_graph, run_graph

# Force local execution by mocking remote client url check
async def mock_get_remote_client_url():
    return None

graph_mod._get_remote_client_url = mock_get_remote_client_url

async def main():
    settings = get_settings()
    graph = build_teknofest_graph(settings)
    
    question = "Blokzincir Yarışması ne"
    print(f"Running query: '{question}'...")
    
    try:
        res = await run_graph(
            graph=graph,
            question=question,
            chat_history=[]
        )
        
        # Write to utf-8 file to avoid terminal encoding issues
        with open("scratch/query_output.txt", "w", encoding="utf-8") as f:
            f.write("=== RESPONSE ===\n")
            f.write(res.get("answer") or "")
            f.write("\n\n=== ROUTE TAKEN ===\n")
            f.write(res.get("route_taken") or "unknown")
            f.write("\n\n=== SOURCES ===\n")
            for src in res.get("sources", []):
                f.write(f"- {src.get('metadata', {}).get('source') or src.get('metadata', {}).get('url') or 'unknown'}\n")
        
        print("\nSuccessfully wrote response to scratch/query_output.txt")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import sys
sys.path.insert(0, ".")
from app.config import get_settings
from app.agent.agent_node import run_agent_node

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    settings = get_settings()
    
    # We simulate the context block retrieved for "Blokzincir Yarışması ne"
    # which did not contain blockchain info
    context_str = "No blockchain information found here."
    
    question = "Blokzincir Yarışması ne"
    
    print("Running run_agent_node...")
    ans = await run_agent_node(
        question=question,
        context_str=context_str,
        settings=settings,
        chat_history=[]
    )
    print("\n--- AGENT ANSWER ---")
    print(ans)

if __name__ == "__main__":
    asyncio.run(main())

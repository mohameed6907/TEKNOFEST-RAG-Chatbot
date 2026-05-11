import asyncio, json, httpx, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = "http://127.0.0.1:8000"

async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/api/auth/guest")
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        queries = [
            "TEKNOFEST nedir?",
            "En sevdiğin kategori?",
            "Hava durumu nasıl?",
        ]

        for q in queries:
            r = await c.post(f"{BASE}/api/chat", headers=headers, json={"message": q})
            data = r.json()
            meta = data.get("meta", {})
            intent = meta.get("intent", "?")
            hal = meta.get("hallucination_check", {})
            local_conf = meta.get("local_confidence", "?")
            reranker = meta.get("reranker_used", "?")
            ctx_count = meta.get("context_chunks_count", "?")
            print(f"Q: {q}")
            print(f"  intent={intent} | route={data.get('route_taken')} | hal={hal.get('status','?')} ({hal.get('reason','')}) | local_conf={local_conf} | ctx_count={ctx_count}")
            print(f"  answer: {data.get('answer','')[:100]}")
            print()

asyncio.run(main())

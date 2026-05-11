import asyncio, json, httpx, sys, io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "http://127.0.0.1:8000"

async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        # 1. Get guest token
        r = await c.post(f"{BASE}/api/auth/guest")
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        print("=== GUEST TOKEN ACQUIRED ===\n")

        session_id = None

        queries = [
            ("TEKNOFEST nedir?", "local"),
            ("En sevdiğin kategori?", "kisisel, zero sources"),
            ("Yarışma hakkında önerin?", "local or llm_knowledge, zero unused sources"),
            ("Hava durumu nasıl?", "diger, zero sources"),
        ]

        for i, (q, expected) in enumerate(queries, 1):
            body = {"message": q}
            if session_id:
                body["session_id"] = session_id
            
            print(f"{'='*60}")
            print(f"QUERY {i}: {q}")
            print(f"EXPECTED: {expected}")
            print(f"{'='*60}")
            
            r = await c.post(f"{BASE}/api/chat", headers=headers, json=body)
            data = r.json()
            
            if not session_id:
                session_id = data.get("session_id")
            
            # Print key fields
            print(f"  route_taken: {data.get('route_taken')}")
            print(f"  sources count: {len(data.get('sources', []))}")
            answer = data.get('answer', '')
            print(f"  answer (first 200): {answer[:200]}")
            if data.get('sources'):
                for si, s in enumerate(data['sources']):
                    src = s.get('metadata', {}).get('source', '') or s.get('type', '')
                    print(f"  source[{si}]: {s.get('type')} | trust={s.get('trust_label','?')} | score={s.get('score')}")
            print()

asyncio.run(main())

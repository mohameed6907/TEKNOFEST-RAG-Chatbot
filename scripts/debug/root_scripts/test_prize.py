import httpx, json, sys, os

BASE = "http://127.0.0.1:8010"

r = httpx.post(f"{BASE}/api/auth/guest")
token = r.json()["access_token"]

r2 = httpx.post(
    f"{BASE}/api/chat",
    json={"message": "İnsansız Kara Aracı yarışmasının ödülü nedir?"},
    headers={"Authorization": f"Bearer {token}"},
    timeout=120.0,
)
with open('test_prize_output2_utf8.json', 'w', encoding='utf-8') as f:
    json.dump(r2.json(), f, indent=2, ensure_ascii=False)

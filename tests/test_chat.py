"""Quick end-to-end chat test."""
import httpx, json, sys, os

# Clean env for direct test runs
if os.getenv("OPENAI_BASE_URL") == "":
    del os.environ["OPENAI_BASE_URL"]

BASE = "http://127.0.0.1:8010"

# 1. Guest login
r = httpx.post(f"{BASE}/api/auth/guest")
print(f"1. Guest Auth: {r.status_code}")
if r.status_code != 200:
    print("   FAIL:", r.text)
    sys.exit(1)
token = r.json()["access_token"]

# 2. Chat
print("2. Sending chat message...")
r2 = httpx.post(
    f"{BASE}/api/chat",
    json={"message": "Drone yarışması başvuru tarihi ne zaman?"},
    headers={"Authorization": f"Bearer {token}"},
    timeout=120.0,
)
print(f"   Chat Status: {r2.status_code}")
d = r2.json()
if r2.status_code == 200:
    print(f"   Answer: {d.get('answer','')[:300]}")
    print(f"   Route: {d.get('route_taken')}")
    print(f"   Sources: {len(d.get('sources', []))}")
else:
    print(f"   Error: {json.dumps(d, indent=2, ensure_ascii=False)}")
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")

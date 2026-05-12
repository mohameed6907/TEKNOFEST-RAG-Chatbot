import asyncio
import json
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, engine
from app.models import Base, User

Base.metadata.create_all(bind=engine)

db = SessionLocal()
user = db.query(User).first()
if not user:
    user = User(email="test@example.com", hashed_password="pw", name="test")
    db.add(user)
    db.commit()

client = TestClient(app)

from app.auth import get_current_user
app.dependency_overrides[get_current_user] = lambda: user

queries = [
    "İKA yarışması için daha fazla bilgi verir misin",
    "parkur hakkında daha fazla bilgi var mı",
    "yarışma parkuru hakkında daha fazla bilgi mevcut mu"
]

session_id = None
results = []

for q in queries:
    payload = {"message": q}
    if session_id:
        payload["session_id"] = session_id
        
    response = client.post(
        "/api/chat",
        json=payload
    )
    
    data = None
    try:
        data = response.json()
        if not session_id and "session_id" in data:
            session_id = data["session_id"]
    except Exception:
        pass
        
    results.append({
        "query": q,
        "status": response.status_code,
        "response": data or response.text
    })

with open("scratch/test_chat_fixes_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Test complete. Results saved to scratch/test_chat_fixes_output.json")

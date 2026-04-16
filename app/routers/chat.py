import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ChatSession, Message, MessageRole, User
from app.auth import get_current_user
from app.rag.graph import run_graph, build_teknofest_graph
from app.config import get_settings

router = APIRouter(prefix="/api/chat", tags=["chat"])
settings = get_settings()

graph = build_teknofest_graph(settings=settings)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: List[dict] = []
    created_at: str
    route_taken: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[dict] = []
    route_taken: str

@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
    return [
        ChatSessionResponse(id=s.id, title=s.title, created_at=s.created_at.isoformat())
        for s in sessions
    ]

@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
def get_messages(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return [
        MessageResponse(
            id=m.id,
            role=m.role.value,
            content=m.content,
            sources=m.sources,
            created_at=m.created_at.isoformat(),
            route_taken=m.route_taken
        ) for m in session.messages
    ]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    return {"status": "success"}

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")

    # Handle session creation or fetch
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        new_session = ChatSession(
            id=session_id,
            user_id=current_user.id,
            title=req.message[:30] + "..." if len(req.message) > 30 else req.message
        )
        db.add(new_session)
        db.commit()
    else:
        # Verify ownership
        session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Save User message
    user_msg = Message(
        session_id=session_id,
        role=MessageRole.USER,
        content=req.message
    )
    db.add(user_msg)
    db.commit()

    # Load history
    history_records = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()
    # Build LangChain format history. We take the last 10 messages for context so we don't blow up context window.
    chat_history = []
    for r in history_records[-11:-1]: # exclude the current user message and take last 10
        role = "user" if r.role == MessageRole.USER else "assistant"
        chat_history.append({"role": role, "content": r.content})

    # Run LangGraph with history
    try:
        result = await run_graph(graph=graph, question=req.message, chat_history=chat_history)
    except Exception as exc:  # pragma: no cover
        # Still need to respond or log
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    route_taken = result.get("route_taken", "unknown")

    # Save AI message
    ai_msg = Message(
        session_id=session_id,
        role=MessageRole.AI,
        content=answer,
        route_taken=route_taken,
        sources_json=json.dumps(sources) if sources else None
    )
    db.add(ai_msg)
    db.commit()

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        route_taken=route_taken
    )

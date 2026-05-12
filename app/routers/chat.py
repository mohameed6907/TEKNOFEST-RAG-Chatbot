import json
import logging
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])
settings = get_settings()

graph = build_teknofest_graph(settings=settings)


def get_langfuse_handler(settings, session_id: str, user_id: str):
    return None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatSessionEditRequest(BaseModel):
    title: str

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

@router.put("/sessions/{session_id}", response_model=ChatSessionResponse)
def edit_session(session_id: str, req: ChatSessionEditRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.title = req.title
    db.commit()
    return ChatSessionResponse(id=session.id, title=session.title, created_at=session.created_at.isoformat())

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
            title=req.message[:250]
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

    # Load history — fetch the most recent messages for rephrase context
    recent = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )

    # Build LangChain format history.
    # Include both user AND assistant messages so the rephrase chain can resolve
    # references to things the bot said (e.g. "parkur" → İKA yarışması parkuru).
    # Truncate assistant content to 200 chars to keep token usage in check.
    chat_history = []
    for m in reversed(recent):
        # Skip the current user message we just inserted
        if m.id == user_msg.id:
            continue
        
        # Standardize roles: user -> user, ai -> assistant
        role = "user" if m.role == MessageRole.USER else "assistant"
        content = m.content
        if role == "assistant" and len(content) > 200:
            content = content[:200] + "..."
        chat_history.append({"role": role, "content": content})

    # No longer using langfuse callback.
    callbacks = []

    # Run LangGraph with history, callbacks, and session metadata
    try:
        result = await run_graph(
            graph=graph,
            question=req.message,
            chat_history=chat_history,
            callbacks=callbacks,
            metadata={
                "session_id": session_id,
                "user_id": str(current_user.id),
                "user_role": current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role),
            },
        )
    except Exception as exc:  # pragma: no cover
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

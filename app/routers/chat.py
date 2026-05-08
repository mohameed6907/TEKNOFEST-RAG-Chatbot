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
    """
    E.3 — Langfuse v4 LangchainCallbackHandler oluşturur.
    langfuse_enabled=False ise None döner.
    v4: Secret/host env var'lardan okunur (LANGFUSE_SECRET_KEY, LANGFUSE_HOST).
    """
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # v4+
        # v4: public_key parametresi opsiyonel; env var'lardan otomatik okunur.
        handler = LangfuseCallbackHandler(
            public_key=settings.langfuse_public_key,
        )
        return handler
    except ImportError:
        logger.warning("langfuse paketi kurulu değil — 'pip install langfuse' çalıştırın.")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse handler oluşturulamadı: %s", exc)
        return None

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
    # IMPORTANT: We include ALL messages up to (but not including) the current user message
    chat_history = []
    for r in history_records[:-1]:  # Exclude only the CURRENT message we just added
        role = "user" if r.role == MessageRole.USER else "assistant"
        chat_history.append({"role": role, "content": r.content})
    
    # Keep only last 10 for token budget (5 turns = 10 messages)
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # E.3 — Langfuse callback (LangSmith ile birlikte çalışabilir)
    callbacks = []
    langfuse_handler = get_langfuse_handler(
        settings, session_id=session_id, user_id=str(current_user.id)
    )
    if langfuse_handler:
        callbacks.append(langfuse_handler)

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

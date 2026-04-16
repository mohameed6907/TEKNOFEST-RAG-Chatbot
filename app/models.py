import enum
from datetime import datetime
import json

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class MessageRole(str, enum.Enum):
    USER = "user"
    AI = "ai"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True) # User's name
    email = Column(String(255), unique=True, index=True, nullable=True) # Null for guest
    hashed_password = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_guest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(50), primary_key=True, index=True) # e.g. UUID
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="Yeni Sohbet")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey("chat_sessions.id"))
    role = Column(Enum(MessageRole))
    content = Column(Text, nullable=False)
    
    # Optional metadata
    route_taken = Column(String(50), nullable=True)
    sources_json = Column(Text, nullable=True) # Store JSON encoded list of sources
    
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
    
    @property
    def sources(self):
        if self.sources_json:
            try:
                return json.loads(self.sources_json)
            except Exception:
                return []
        return []

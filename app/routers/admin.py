import os
import shutil
import subprocess
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, ChatSession
from app.auth import get_admin_user
from app.config import get_settings

router = APIRouter(prefix="/api/admin", tags=["admin"])
settings = get_settings()

class ConfigUpdate(BaseModel):
    llm_provider: str
    retrieval_top_k: int
    enable_reranking: bool
    rag_confidence_threshold: float

@router.get("/config")
def get_config(admin: User = Depends(get_admin_user)):
    return {
        "llm_provider": settings.llm_provider,
        "retrieval_top_k": settings.retrieval_top_k,
        "enable_reranking": settings.reranker_enabled,
        "rag_confidence_threshold": settings.rag_confidence_threshold,
    }

@router.post("/config")
def update_config(config: ConfigUpdate, admin: User = Depends(get_admin_user)):
    env_path = settings.base_dir / ".env"
    
    # Simple env update
    updates = {
        "LLM_PROVIDER": config.llm_provider,
        "RETRIEVAL_TOP_K": str(config.retrieval_top_k),
        "ENABLE_RERANKING": "true" if config.enable_reranking else "false",
        "RAG_CONFIDENCE_THRESHOLD": str(config.rag_confidence_threshold)
    }
    
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        with open(env_path, "w") as f:
            for line in lines:
                written = False
                for k, v in updates.items():
                    if line.startswith(k + "="):
                        f.write(f"{k}={v}\n")
                        del updates[k]
                        written = True
                        break
                if not written:
                    f.write(line)
            
            # Append any remaining new keys
            for k, v in updates.items():
                f.write(f"{k}={v}\n")
    else:
        with open(env_path, "w") as f:
            for k, v in updates.items():
                f.write(f"{k}={v}\n")
                
    return {"status": "success", "message": "Configuration updated. Application partial reload may be required."}


@router.post("/ingest")
def trigger_ingestion(admin: User = Depends(get_admin_user)):
    script_path = settings.base_dir / "scripts" / "ingest_local_docs.py"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Ingestion script not found")
        
    try:
        # Run natively in background or block
        # For simplicity in this demo, we run synchronously
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(settings.base_dir)
        )
        return {"status": "success", "output": result.stdout, "errors": result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
def get_users(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    users = db.query(User).all()
    user_list = []
    for u in users:
        session_count = db.query(ChatSession).filter(ChatSession.user_id == u.id).count()
        user_list.append({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "role": u.role.value,
            "is_guest": u.is_guest,
            "session_count": session_count,
            "created_at": u.created_at.isoformat() if u.created_at else None
        })
    return user_list

@router.get("/files")
def get_files(admin: User = Depends(get_admin_user)):
    raw_dir = settings.rag_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    file_list = []
    for file_path in raw_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            stat = file_path.stat()
            file_list.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
            
    # Sort files by newest first
    file_list.sort(key=lambda x: x["created_at"], reverse=True)
    return file_list

@router.post("/files")
def upload_file(file: UploadFile = File(...), admin: User = Depends(get_admin_user)):
    raw_dir = settings.rag_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = raw_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename, "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@router.delete("/files/{filename}")
def delete_file(filename: str, admin: User = Depends(get_admin_user)):
    raw_dir = settings.rag_root / "raw"
    file_path = raw_dir / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        file_path.unlink()
        return {"status": "success", "message": f"File {filename} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


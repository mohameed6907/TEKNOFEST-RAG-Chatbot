from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import get_settings
from .rag.graph import build_teknofest_graph, run_graph
from .tracing import init_langsmith, is_tracing_enabled
from .database import engine, Base

# Import routers
from .routers import auth, chat, admin

# Initialize DB tables
Base.metadata.create_all(bind=engine)


def create_app() -> FastAPI:
    settings = get_settings()

    # Initialise LangSmith tracing BEFORE building the graph so every
    # LangChain / LangGraph call is captured from the very first request.
    init_langsmith(settings)

    app = FastAPI(title=settings.app_name)

    # Static & templates
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / "static"
    templates_dir = base_dir / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # Build LangGraph workflow once at startup
    # Note: we are not storing `graph` globally anymore because `app/routers/chat.py` builds it independently,
    # but we can leave it here if other parts use it.

    # Include routers
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "tracing_enabled": is_tracing_enabled(),
            "langsmith_project": settings.langsmith_project if is_tracing_enabled() else None,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def index_fallback(request: Request, full_path: str) -> HTMLResponse:
        # Avoid intercepting API calls or static files
        if full_path.startswith("api/") or full_path.startswith("static/"):
            raise HTTPException(status_code=404)
        return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})

    return app


app = create_app()


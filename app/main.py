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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    route_taken: str
    meta: Dict[str, Any] = {}


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name)

    # Static & templates
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / "static"
    templates_dir = base_dir / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # Build LangGraph workflow once at startup
    graph = build_teknofest_graph(settings=settings)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> JSONResponse:
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="message is empty")

        try:
            result = await run_graph(graph=graph, question=req.message)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(
            ChatResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                route_taken=result.get("route_taken", "unknown"),
                meta=result.get("meta", {}),
            ).model_dump()
        )

    return app


app = create_app()


from __future__ import annotations

from fastapi import FastAPI

from .agent import RecommendationAgent
from .catalog import Catalog
from .models import ChatRequest, ChatResponse

app = FastAPI()
_catalog = Catalog.load()
_agent = RecommendationAgent(_catalog)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _agent.respond(request.messages)

from __future__ import annotations

from fastapi import FastAPI

from .agent import RecommendationAgent
from .catalog import Catalog
from .models import ChatRequest, ChatResponse

app = FastAPI()
_catalog = Catalog.load()
_agent = RecommendationAgent(_catalog)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "message": "SHL recommendation API is running.",
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/health",
            },
            "chat": {
                "method": "POST",
                "path": "/chat",
                "body": {
                    "messages": [
                        {"role": "user", "content": "I need assessments for a Java developer."}
                    ]
                },
            },
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chat")
def chat_get_help() -> dict[str, object]:
    return {
        "error": "Method Not Allowed for this route with GET.",
        "how_to_use": {
            "method": "POST",
            "path": "/chat",
            "body": {
                "messages": [
                    {"role": "user", "content": "I need assessments for a Java developer."}
                ]
            },
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _agent.respond(request.messages)

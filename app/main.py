from __future__ import annotations


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import RecommendationAgent
from .catalog import Catalog
from .models import ChatRequest, ChatResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_catalog = Catalog.load()
_agent = RecommendationAgent(_catalog)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _agent.respond(request.messages)

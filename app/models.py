from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class Assessment(BaseModel):
    name: str
    url: str
    test_types: list[str]
    remote_testable: bool = False
    adaptive: bool = False
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    assessment_length: str = ""
    downloads: list[dict[str, str]] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    searchable_text: str = ""

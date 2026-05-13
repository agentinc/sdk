from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = Field(default_factory=list)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AgentInput(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

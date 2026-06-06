from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

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


# ---------------------------------------------------------------------------
# Configuration TypedDicts — passed as plain dicts by developers
# ---------------------------------------------------------------------------

class ModelConfig(TypedDict, total=False):
    model: Required[str]      # e.g. "gpt-4o-mini", "claude-sonnet-4-6", "gemini-1.5-pro"
    api_key: Required[str]
    base_url: str             # optional — enables any OpenAI-compatible endpoint


class MemoryConfig(TypedDict, total=False):
    type: Required[Literal["redis"]]
    connection: Required[str]  # e.g. "redis://localhost:6379/0"
    user: str
    password: str


class MCPStdioConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: list[str]


class MCPSseConfig(TypedDict):
    type: Literal["sse"]
    url: str


MCPConfig = MCPStdioConfig | MCPSseConfig


class DataConfig(TypedDict, total=False):
    """Reserved for native RAG — accepted but not yet implemented."""
    type: str
    storage_dir: str
    mode: str

"""Canonical message format (v0.2 WP-3).

The runtime stores and operates on this format end-to-end. Provider-specific
wire formats are produced + parsed at the edges by ``adapter/wire``.

Every LLM call goes:

    history (CanonicalMessage[]) ---adapter/wire.toWire---> provider request
    provider response ---adapter/wire.fromWire---> CanonicalChunk stream

Stored messages (postgres ``Message.content``) always serialize ``ContentBlock[]``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

CanonicalRole = Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Content blocks — the only shapes that may appear inside CanonicalMessage.content
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source_url: str | None = None
    media_type: str | None = None
    data_b64: str | None = None


class ToolUseBlock(BaseModel):
    """An assistant turn requesting a tool call."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """A tool turn returning the result of a previous tool_use."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: list["ContentBlock"] = Field(default_factory=list)
    is_error: bool = False


class ThinkingBlock(BaseModel):
    """Provider-emitted reasoning trace (Anthropic ``thinking``, OpenAI ``reasoning``)."""

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


ContentBlock = Annotated[
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]

# Late-bind forward refs.
ToolResultBlock.model_rebuild()


# ---------------------------------------------------------------------------
# Canonical message
# ---------------------------------------------------------------------------


class CanonicalMessage(BaseModel):
    """One turn in the canonical conversation history."""

    role: CanonicalRole
    content: list[ContentBlock]
    from_agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming chunks emitted by adapter/wire
# ---------------------------------------------------------------------------


class ChunkText(BaseModel):
    type: Literal["text"] = "text"
    delta: str


class ChunkThinking(BaseModel):
    type: Literal["thinking"] = "thinking"
    delta: str


class ChunkToolUseStart(BaseModel):
    type: Literal["tool_use_start"] = "tool_use_start"
    id: str
    name: str


class ChunkToolUseInputDelta(BaseModel):
    type: Literal["tool_use_input_delta"] = "tool_use_input_delta"
    id: str
    delta: str


class ChunkToolUseStop(BaseModel):
    type: Literal["tool_use_stop"] = "tool_use_stop"
    id: str


class ChunkMessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"] | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class ChunkError(BaseModel):
    type: Literal["error"] = "error"
    message: str
    code: str | None = None


CanonicalChunk = Annotated[
    ChunkText
    | ChunkThinking
    | ChunkToolUseStart
    | ChunkToolUseInputDelta
    | ChunkToolUseStop
    | ChunkMessageStop
    | ChunkError,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Tool descriptors carried in InvokeOpts
# ---------------------------------------------------------------------------


class ToolDescriptor(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class InvokeOpts(BaseModel):
    tools: list[ToolDescriptor] = Field(default_factory=list)
    system_prompt: str | None = None
    stream: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


__all__ = [
    "CanonicalRole",
    "TextBlock",
    "ImageBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ContentBlock",
    "CanonicalMessage",
    "ChunkText",
    "ChunkThinking",
    "ChunkToolUseStart",
    "ChunkToolUseInputDelta",
    "ChunkToolUseStop",
    "ChunkMessageStop",
    "ChunkError",
    "CanonicalChunk",
    "ToolDescriptor",
    "InvokeOpts",
]

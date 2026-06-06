"""WP-3: canonical message format — round-trip + discriminator sanity checks."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agentinc.sdk import (
    CanonicalChunk,
    CanonicalMessage,
    ChunkMessageStop,
    ChunkText,
    ChunkToolUseStart,
    ContentBlock,
    ImageBlock,
    InvokeOpts,
    TextBlock,
    ThinkingBlock,
    ToolDescriptor,
    ToolResultBlock,
    ToolUseBlock,
)


def test_text_block_round_trip() -> None:
    raw = {"type": "text", "text": "hello"}
    block = TypeAdapter(ContentBlock).validate_python(raw)
    assert isinstance(block, TextBlock)
    assert block.model_dump() == raw


def test_tool_use_then_tool_result_pair() -> None:
    msg_assistant = CanonicalMessage(
        role="assistant",
        content=[
            TextBlock(text="thinking out loud"),
            ToolUseBlock(id="tu_1", name="search", input={"q": "weather"}),
        ],
    )
    msg_tool = CanonicalMessage(
        role="tool",
        content=[
            ToolResultBlock(
                tool_use_id="tu_1",
                content=[TextBlock(text="73F sunny")],
            )
        ],
    )
    history = [msg_assistant, msg_tool]
    dumped = [m.model_dump() for m in history]
    rehydrated = [CanonicalMessage.model_validate(d) for d in dumped]
    assert rehydrated == history


def test_thinking_block_optional_signature() -> None:
    b = ThinkingBlock(text="reasoning…")
    assert b.signature is None
    b2 = ThinkingBlock(text="signed reasoning", signature="abc")
    assert b2.signature == "abc"


def test_image_block_allows_url_or_inline() -> None:
    ImageBlock(source_url="https://example.com/cat.png", media_type="image/png")
    ImageBlock(data_b64="ZmFrZQ==", media_type="image/png")


def test_unknown_block_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ContentBlock).validate_python({"type": "video", "url": "x"})


def test_canonical_chunk_discriminator() -> None:
    adapter = TypeAdapter(CanonicalChunk)
    assert isinstance(adapter.validate_python({"type": "text", "delta": "hi"}), ChunkText)
    assert isinstance(
        adapter.validate_python({"type": "tool_use_start", "id": "tu_1", "name": "search"}),
        ChunkToolUseStart,
    )
    assert isinstance(
        adapter.validate_python(
            {"type": "message_stop", "stop_reason": "end_turn", "tokens_in": 12, "tokens_out": 3}
        ),
        ChunkMessageStop,
    )


def test_invoke_opts_defaults() -> None:
    opts = InvokeOpts()
    assert opts.stream is True
    assert opts.tools == []
    assert opts.system_prompt is None


def test_invoke_opts_with_tools() -> None:
    tool = ToolDescriptor(
        name="get_weather",
        description="Fetch current weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    opts = InvokeOpts(tools=[tool], system_prompt="be helpful", max_tokens=512, temperature=0.2)
    assert opts.tools[0].name == "get_weather"
    assert opts.system_prompt == "be helpful"


def test_role_is_constrained() -> None:
    with pytest.raises(ValidationError):
        CanonicalMessage(role="banana", content=[TextBlock(text="x")])  # type: ignore[arg-type]

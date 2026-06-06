from __future__ import annotations

import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.anthropic")

_SYSTEM_KEY = "__system__"


def _strip_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull the system message out; Anthropic takes it as a separate param."""
    system = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            system = m.get("content", "")
        else:
            rest.append(m)
    return system, rest


class AnthropicProvider:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic extra: "
                "pip install 'agentinc-sdk[anthropic]'"
            )
        self._client = self._anthropic.AsyncAnthropic(api_key=config["api_key"])
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        system, msgs = _strip_system(messages)

        anthropic_tools = (
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
            if tools
            else []
        )

        kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            messages=msgs,
        )
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if stream:
            async for chunk in self._stream(kwargs):
                yield chunk
        else:
            async for chunk in self._blocking(kwargs):
                yield chunk

    async def _stream(self, kwargs: dict) -> AsyncIterator[AgentOutput]:
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield AgentOutput(content=text, done=False)

            msg = await stream.get_final_message()

        if msg.stop_reason == "tool_use":
            tool_calls = []
            for block in msg.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                    )
            yield AgentOutput(tool_calls=tool_calls, done=False)
        else:
            yield AgentOutput(content="", done=True)

    async def _blocking(self, kwargs: dict) -> AsyncIterator[AgentOutput]:
        msg = await self._client.messages.create(**kwargs)

        if msg.stop_reason == "tool_use":
            tool_calls = []
            for block in msg.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                    )
            yield AgentOutput(tool_calls=tool_calls, done=False)
        else:
            text = "".join(b.text for b in msg.content if hasattr(b, "text"))
            yield AgentOutput(content=text, done=True)

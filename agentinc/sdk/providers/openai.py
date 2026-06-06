from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.openai")


class OpenAIProvider:
    """Covers OpenAI and any OpenAI-compatible endpoint (DeepSeek, Groq, Ollama, …)."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai extra: "
                "pip install 'agentinc-sdk[openai]'"
            )
        self._client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
        )
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        openai_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tools
            else None
        )

        if stream:
            async for chunk in self._stream(messages, openai_tools):
                yield chunk
        else:
            async for chunk in self._blocking(messages, openai_tools):
                yield chunk

    async def _stream(self, messages: list[dict], openai_tools) -> AsyncIterator[AgentOutput]:
        accumulated: dict[int, dict] = {}

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield AgentOutput(content=delta.content, done=False)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated:
                        accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulated[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            accumulated[idx]["arguments"] += tc.function.arguments

            if choice.finish_reason == "tool_calls":
                tool_calls = [
                    ToolCall(
                        id=v["id"],
                        name=v["name"],
                        arguments=json.loads(v["arguments"]) if v["arguments"] else {},
                    )
                    for v in accumulated.values()
                ]
                yield AgentOutput(tool_calls=tool_calls, done=False)
                return

            if choice.finish_reason == "stop":
                yield AgentOutput(content="", done=True)
                return

    async def _blocking(self, messages: list[dict], openai_tools) -> AsyncIterator[AgentOutput]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            stream=False,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            yield AgentOutput(tool_calls=tool_calls, done=False)
        else:
            yield AgentOutput(content=msg.content or "", done=True)

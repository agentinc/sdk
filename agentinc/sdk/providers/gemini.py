from __future__ import annotations

import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.gemini")


def _to_gemini_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Gemini contents + system instruction."""
    system = ""
    contents = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            system = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [{"function_response": {
                    "name": m.get("name", ""),
                    "response": {"result": content},
                }}],
            })
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    return system, contents


class GeminiProvider:
    def __init__(self, config: ModelConfig) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self._types = genai_types
        except ImportError:
            raise ImportError(
                "Gemini provider requires the gemini extra: "
                "pip install 'agentinc-sdk[gemini]'"
            )
        self._client = genai.Client(api_key=config["api_key"])
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        system, contents = _to_gemini_messages(messages)

        gemini_tools = None
        if tools:
            function_declarations = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
                for t in tools
            ]
            gemini_tools = [self._types.Tool(function_declarations=function_declarations)]

        config_kwargs: dict = {}
        if system:
            config_kwargs["system_instruction"] = system
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        gen_config = self._types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if stream:
            async for chunk in self._stream(contents, gen_config):
                yield chunk
        else:
            async for chunk in self._blocking(contents, gen_config):
                yield chunk

    async def _stream(self, contents, gen_config) -> AsyncIterator[AgentOutput]:
        kwargs = {"model": self._model, "contents": contents}
        if gen_config:
            kwargs["config"] = gen_config

        async for chunk in await self._client.aio.models.generate_content_stream(**kwargs):
            if chunk.function_calls:
                tool_calls = [
                    ToolCall(id=fc.id or fc.name, name=fc.name, arguments=dict(fc.args or {}))
                    for fc in chunk.function_calls
                ]
                yield AgentOutput(tool_calls=tool_calls, done=False)
                return
            if chunk.text:
                yield AgentOutput(content=chunk.text, done=False)

        yield AgentOutput(content="", done=True)

    async def _blocking(self, contents, gen_config) -> AsyncIterator[AgentOutput]:
        kwargs = {"model": self._model, "contents": contents}
        if gen_config:
            kwargs["config"] = gen_config

        response = await self._client.aio.models.generate_content(**kwargs)

        if response.function_calls:
            tool_calls = [
                ToolCall(id=fc.id or fc.name, name=fc.name, arguments=dict(fc.args or {}))
                for fc in response.function_calls
            ]
            yield AgentOutput(tool_calls=tool_calls, done=False)
        else:
            yield AgentOutput(content=response.text or "", done=True)

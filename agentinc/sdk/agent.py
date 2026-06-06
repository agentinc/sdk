from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any, AsyncIterator

from .memory import memory_for
from .memory.base import Memory
from .providers import provider_for
from .providers.base import Provider
from .schemas import (
    AgentInput,
    AgentOutput,
    DataConfig,
    MCPConfig,
    MemoryConfig,
    Message,
    ModelConfig,
    ToolCall,
    ToolSchema,
)
from .tool import ToolWrapper, _build_schema

log = logging.getLogger("agentinc.sdk.agent")


def _wrap_tool(fn: Callable) -> ToolWrapper:
    if isinstance(fn, ToolWrapper):
        return fn
    schema = _build_schema(fn, name=fn.__name__, description=fn.__doc__ or "")
    return ToolWrapper(fn=fn, schema=schema)


class Agent:
    """
    High-level agent that wires together a provider, tools, MCP servers,
    Redis memory, and optional context into a single AgentProtocol-compatible object.

    Usage::

        agent = Agent(
            role="You are a helpful assistant.",
            model={"model": "gpt-4o-mini", "api_key": "sk-…"},
            tools=[my_function],
            memory={"type": "redis", "connection": "redis://localhost:6379"},
        )
        serve(agent, name="my-agent", port=8000)
    """

    def __init__(
        self,
        role: str,
        model: ModelConfig,
        tools: list[Callable] | None = None,
        mcps: list[MCPConfig] | None = None,
        memory: MemoryConfig | None = None,
        context: str | None = None,
        data: DataConfig | None = None,
    ) -> None:
        self._role = role
        self._context = context
        self._model_config = model

        self._provider: Provider = provider_for(model)
        self._memory: Memory | None = memory_for(memory) if memory else None
        self._mcp_configs: list[MCPConfig] = mcps or []

        # Wrap plain functions as ToolWrapper instances
        self._local_tools: dict[str, ToolWrapper] = {
            w.schema().name: w for w in (_wrap_tool(t) for t in (tools or []))
        }

        # MCP tool registry — populated on first run()
        self._mcp_tools: dict[str, Any] = {}
        self._mcp_ready = False

        if data is not None:
            log.debug("data= parameter accepted but not yet implemented (reserved for RAG)")

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        if self._mcp_configs and not self._mcp_ready:
            await self._init_mcp()

        session_id = input.metadata.get("session_id") or str(uuid.uuid4())

        # Load history from memory backend (empty list if no memory configured)
        persisted: list[Message] = []
        if self._memory:
            persisted = await self._memory.load(session_id)

        # Merge caller-supplied history (takes precedence) with persisted history
        history = persisted if not input.history else list(input.history)

        # Build system prompt
        system_parts = [self._role]
        if self._context:
            system_parts.append(self._context)

        messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for msg in history:
            messages.append(self._message_to_dict(msg))
        messages.append({"role": "user", "content": input.message})

        # Collect all tool schemas
        tool_schemas = [w.schema() for w in self._local_tools.values()]
        tool_schemas += [
            ToolSchema(name=name, description=spec["description"], parameters=spec["parameters"])
            for name, spec in self._mcp_tools.items()
        ]

        # Agentic loop
        new_messages: list[dict] = []
        while True:
            tool_calls_batch: list[ToolCall] = []

            async for chunk in self._provider.complete(messages + new_messages, tool_schemas):
                if chunk.tool_calls:
                    tool_calls_batch.extend(chunk.tool_calls)
                elif chunk.content:
                    yield chunk
                if chunk.done:
                    break

            if not tool_calls_batch:
                yield AgentOutput(content="", done=True)
                break

            # Append assistant tool-call turn
            new_messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in tool_calls_batch
                ],
            })

            # Dispatch all tool calls and append results
            for tc in tool_calls_batch:
                result = await self._dispatch_tool(tc)
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Persist updated history
        if self._memory:
            updated = list(history) + [
                Message(role="user", content=input.message),
                *[
                    Message(role=m["role"], content=m.get("content", ""))
                    for m in new_messages
                    if m["role"] in ("assistant", "tool") and m.get("content")
                ],
            ]
            await self._memory.save(session_id, updated)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, tc: ToolCall) -> str:
        if tc.name in self._local_tools:
            return await self._local_tools[tc.name].call(tc)
        if tc.name in self._mcp_tools:
            return await self._call_mcp_tool(tc)
        return f"Error: tool '{tc.name}' not found"

    async def _init_mcp(self) -> None:
        for config in self._mcp_configs:
            try:
                tools = await _fetch_mcp_tools(config)
                self._mcp_tools.update(tools)
            except Exception:
                log.exception("Failed to initialise MCP server: %s", config)
        self._mcp_ready = True

    async def _call_mcp_tool(self, tc: ToolCall) -> str:
        spec = self._mcp_tools.get(tc.name)
        if spec is None:
            return f"Error: MCP tool '{tc.name}' not found"
        try:
            result = await _invoke_mcp_tool(spec["config"], tc.name, tc.arguments)
            return result
        except Exception as exc:
            return f"Error calling MCP tool '{tc.name}': {exc}"

    @staticmethod
    def _message_to_dict(msg: Message) -> dict:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        return d


# ------------------------------------------------------------------
# MCP helpers (lazy-import mcp package)
# ------------------------------------------------------------------

async def _fetch_mcp_tools(config: MCPConfig) -> dict[str, dict]:
    """Connect to an MCP server and return its tool schemas."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client
    except ImportError:
        raise ImportError(
            "MCP support requires the mcp extra: pip install 'agentinc-sdk[mcp]'"
        )

    tools: dict[str, dict] = {}

    if config["type"] == "stdio":
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    elif config["type"] == "sse":
        async with sse_client(config["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    return tools


async def _invoke_mcp_tool(config: MCPConfig, name: str, arguments: dict) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client

    if config["type"] == "stdio":
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return str(result.content)

    elif config["type"] == "sse":
        async with sse_client(config["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return str(result.content)

    return "Error: unsupported MCP transport"

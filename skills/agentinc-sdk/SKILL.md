---
name: agentinc-sdk
description: "How to build, wrap, and serve AI agents using the agentinc-sdk Python package. Use this skill whenever the user is working with agentinc, agentinc-sdk, AgentProtocol, RawAdapter, or wants to create agents for the Agentinc marketplace. Also trigger when you see imports from agentinc.sdk, files referencing agentinc agent patterns, or when the user asks about wrapping OpenAI/Anthropic/LangChain/CrewAI agents into a universal protocol. Even if the user doesn't say 'agentinc' explicitly, trigger if they're working in a project that has agentinc-sdk as a dependency or has agentinc/ in its import paths."
---

# Agentinc SDK

The agentinc-sdk is the developer interface for the Agentinc agent marketplace. Developers build agents with any LLM framework, wrap them in a universal protocol, and serve them over A2A (Agent-to-Agent) — all with this single package.

The SDK is **open-source and self-contained** — it depends only on `pydantic>=2.7`.

## Installation

```bash
pip install agentinc-sdk            # core (pydantic only)
pip install agentinc-sdk[serve]     # adds FastAPI + Uvicorn for A2A serving
```

Requires **Python 3.12+**.

## The Core Pattern

Every agent in the Agentinc ecosystem follows this flow:

```
Your function (any signature) → RawAdapter → AgentProtocol → serve()
```

1. Write your agent as a plain Python function (sync, async, streaming — whatever you want)
2. Wrap it with `RawAdapter` which auto-detects the signature and adapts it
3. Optionally serve it over A2A with `serve()` or `create_app()`

This means you **never need to implement AgentProtocol directly** unless you want fine-grained control. RawAdapter handles the translation.

## Quick Reference

### All Exports

```python
from agentinc.sdk import (
    AgentProtocol,     # Protocol — implement run(input) -> AsyncIterator[AgentOutput]
    ToolProtocol,      # Protocol — implement schema() + call()
    AgentFactory,      # Type alias — callable returning AgentProtocol
    AgentInput,        # Pydantic model — agent invocation input
    AgentOutput,       # Pydantic model — single output chunk
    Message,           # Pydantic model — conversation history entry
    ToolCall,          # Pydantic model — tool invocation request
    ToolSchema,        # Pydantic model — tool JSON Schema description
    ToolWrapper,       # Class — wraps callable as ToolProtocol
    tool,              # Decorator — function → ToolWrapper
    RawAdapter,        # Class — wraps any callable as AgentProtocol
)

# Serve module (requires [serve] extra)
from agentinc.sdk.serve import create_app, serve
```

### Minimal Agent (5 lines)

```python
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

async def my_agent(message: str) -> str:
    return f"You said: {message}"

serve(RawAdapter(my_agent), name="echo", port=8000)
```

## Schemas

All models are Pydantic v2 `BaseModel` subclasses.

### AgentInput

The input to every agent invocation.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `message` | `str` | *required* | The user's message |
| `history` | `list[Message]` | `[]` | Conversation history |
| `tool_schemas` | `list[ToolSchema]` | `[]` | Available tools |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

### AgentOutput

A single chunk yielded by an agent. Agents yield one or more of these.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `content` | `str \| None` | `None` | Text content |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool calls the agent wants to make |
| `done` | `bool` | `False` | Whether this is the final chunk |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary output metadata |

### Message

```python
Message(role="user" | "assistant" | "tool", content="...", tool_call_id=None, tool_calls=[])
```

### ToolCall

```python
ToolCall(id="call-1", name="search", arguments={"query": "test"})
```

### ToolSchema

```python
ToolSchema(name="search", description="Searches the web", parameters={"type": "object", ...})
```

## RawAdapter

The most important class in the SDK. It wraps **any** callable as an `AgentProtocol`, auto-detecting the signature pattern.

### Supported Signatures

RawAdapter inspects your function and maps arguments automatically. Your function does **not** need to import anything from agentinc:

| Signature | What happens |
|-----------|-------------|
| `fn(message: str) -> str` | Simple request → response |
| `fn(message: str) -> AsyncIterator[str]` | Streaming text chunks |
| `fn(message: str, history: list) -> ...` | Gets conversation history as plain dicts |
| `fn(message: str, history: list, tools: list) -> ...` | Gets history + tool schemas as dicts |
| `fn(input: AgentInput) -> AgentOutput` | Full control with SDK types |
| `fn(input: AgentInput) -> AsyncIterator[AgentOutput]` | Full streaming with SDK types |

The parameter name `message` must be the first parameter for auto-detection to work. For full control, accept `AgentInput` directly.

### Tool Call Dispatch

When your function yields a dict with `tool_calls`, RawAdapter converts it to a proper `AgentOutput`:

```python
async def my_agent(message: str, history: list, tools: list):
    # Signal that you want to call a tool
    yield {
        "tool_calls": [{"id": "1", "name": "search", "arguments": {"q": message}}],
        "done": False,
    }
```

Read `references/raw-adapter.md` for advanced patterns and edge cases.

## @tool Decorator

Turns any function into a `ToolProtocol` with auto-generated JSON Schema from type hints.

```python
from agentinc.sdk import tool, ToolCall

@tool
async def search(query: str, limit: int = 10) -> str:
    return f"Results for {query}"

# With explicit name and description
@tool(name="add", description="Adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)
```

The resulting `ToolWrapper` can be called two ways:

```python
# Via ToolCall (how the platform dispatches)
result = await search.call(ToolCall(id="1", name="search", arguments={"query": "test"}))

# Direct call
result = await search(query="test")
```

### Type Mappings

| Python | JSON Schema |
|--------|------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list` | `{"type": "array"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `dict` | `{"type": "object"}` |

Parameters without defaults become `required` in the schema.

## AgentProtocol

The universal contract. It's a `runtime_checkable` Protocol — you satisfy it by structural subtyping (just implement the method, no inheritance needed):

```python
@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

Direct implementation (rarely needed — prefer RawAdapter):

```python
from agentinc.sdk import AgentInput, AgentOutput

class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Got: {input.message}", done=True)

assert isinstance(MyAgent(), AgentProtocol)  # passes
```

## Serving Over A2A

The `serve` module exposes agents over HTTP using the A2A protocol (JSON-RPC 2.0 + SSE streaming). Requires the `[serve]` extra.

### serve() — One-liner for local dev

```python
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

serve(RawAdapter(my_fn), name="my-agent", description="Does things", host="0.0.0.0", port=8000)
```

### create_app() — For custom deployment

Returns a FastAPI app you can run with any ASGI server:

```python
from agentinc.sdk.serve import create_app

app = create_app(RawAdapter(my_fn), name="my-agent")
# Run with: uvicorn myfile:app --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/.well-known/agent.json` | Agent card |
| `POST` | `/` | JSON-RPC 2.0 (`tasks/send`, `tasks/sendSubscribe`) |

### Testing with curl

```bash
# Request/response
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello"}]}}}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello"}]}}}'
```

## Framework Integration Patterns

The SDK wraps agents from any framework. The pattern is always:

1. Use the framework's native client/API
2. Convert the framework's response to either a `str` yield or a `dict` with `tool_calls`
3. Wrap with `RawAdapter`
4. Serve with `serve()`

Read `references/frameworks.md` for complete examples for OpenAI, Anthropic, LangChain, and CrewAI.

### Quick Pattern — OpenAI

```python
from openai import AsyncOpenAI
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

client = AsyncOpenAI()

async def openai_agent(message: str, history: list, tools: list):
    messages = [{"role": "user", "content": message}]
    response = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    yield response.choices[0].message.content

serve(RawAdapter(openai_agent), name="openai-agent", port=8000)
```

### Quick Pattern — Anthropic

```python
from anthropic import AsyncAnthropic
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

client = AsyncAnthropic()

async def anthropic_agent(message: str):
    response = await client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    )
    yield response.content[0].text

serve(RawAdapter(anthropic_agent), name="claude-agent", port=8000)
```

## Key Rules

1. **SDK imports only from itself.** Never add imports from `agentinc.core`, `agentinc.runner`, `agentinc.loader`, `agentinc.engine`, or `agentinc.protocols`. Those are platform internals.
2. **RawAdapter is the default path.** Only implement `AgentProtocol` directly if you need something RawAdapter can't provide.
3. **Python 3.12+ required.** Always use `--python 3.12` when creating venvs with `uv`.
4. **The `[serve]` extra is optional.** Core SDK functionality works without FastAPI/Uvicorn.
5. **Tool functions return `str`.** The `@tool` decorator and `ToolWrapper.call()` always return `str`.

## Reference Files

For detailed API docs, read these files as needed:

- `references/api.md` — Complete field-level reference for every schema and protocol
- `references/raw-adapter.md` — All RawAdapter signature patterns, edge cases, and streaming details
- `references/frameworks.md` — Full examples for OpenAI, Anthropic, LangChain, and CrewAI with tool support
- `references/serve.md` — A2A serve module: endpoints, JSON-RPC methods, SSE streaming format

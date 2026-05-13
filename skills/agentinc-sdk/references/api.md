# Agentinc SDK — Complete API Reference

## Table of Contents

1. [Schemas](#schemas)
2. [Protocols](#protocols)
3. [ToolWrapper](#toolwrapper)
4. [RawAdapter](#rawadapter)
5. [Serve Module](#serve-module)

---

## Schemas

All schemas are Pydantic v2 `BaseModel` subclasses from `agentinc.sdk.schemas`.

### AgentInput

```python
class AgentInput(BaseModel):
    message: str                                          # required
    history: list[Message] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- `message` — The user's text input. Always present, never empty.
- `history` — Previous conversation turns as `Message` objects. Empty on first turn.
- `tool_schemas` — Tools available to the agent for this invocation. The platform populates this; developers read it.
- `metadata` — Extensible dict for arbitrary data.

### AgentOutput

```python
class AgentOutput(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- `content` — Text to send to the user. `None` when the output is a tool call with no text.
- `tool_calls` — Tools the agent wants the platform to execute. Mutually exclusive with `done=True` in practice.
- `done` — Signals the final chunk. The platform stops reading after `done=True`.
- `metadata` — Extensible. Use for custom data like token counts, confidence scores, etc.

### Message

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
```

- `role` — Who sent the message.
- `content` — The message text.
- `tool_call_id` — Set when `role="tool"` to link back to the ToolCall that produced this result.
- `tool_calls` — Present when `role="assistant"` and the assistant requested tool execution.

### ToolCall

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
```

- `id` — Unique identifier. Used to correlate tool results back to the call.
- `name` — Must match a `ToolSchema.name` from the available tools.
- `arguments` — Key-value pairs matching the tool's JSON Schema `parameters`.

### ToolSchema

```python
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
```

- `name` — Unique tool identifier.
- `description` — Human-readable description. LLMs use this to decide when to call the tool.
- `parameters` — JSON Schema object describing the tool's accepted arguments.

---

## Protocols

Both protocols are `@runtime_checkable`, meaning `isinstance()` checks work at runtime.

### AgentProtocol

```python
@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

Any object with a `run` method matching this signature satisfies the protocol. No inheritance needed — this is structural subtyping.

### ToolProtocol

```python
@runtime_checkable
class ToolProtocol(Protocol):
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
```

- `schema()` — Returns the tool's JSON Schema description. Called once at registration time.
- `call(tool_call)` — Executes the tool. Always returns `str`.

### AgentFactory

```python
AgentFactory = Any  # callable that receives tools and returns AgentProtocol
```

Type alias for a callable that the platform invokes to create agent instances. Used internally by the platform's loader.

---

## ToolWrapper

```python
class ToolWrapper:
    def __init__(self, fn: Callable[..., Any], schema: ToolSchema) -> None: ...
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
    async def __call__(self, **kwargs: Any) -> str: ...
```

Created by the `@tool` decorator. Satisfies `ToolProtocol`.

- `schema()` — Returns the auto-generated `ToolSchema`.
- `call(tool_call)` — Executes via a `ToolCall` object (platform dispatch path).
- `__call__(**kwargs)` — Direct invocation with keyword arguments.

Both `call()` and `__call__()` handle sync and async functions transparently.

---

## RawAdapter

```python
class RawAdapter:
    def __init__(self, fn: Any) -> None: ...
    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

Satisfies `AgentProtocol`. See `references/raw-adapter.md` for full signature detection rules.

---

## Serve Module

Exported from `agentinc.sdk.serve`. Requires the `[serve]` extra.

### create_app()

```python
def create_app(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
) -> FastAPI: ...
```

Returns a FastAPI application. Use this when you need control over the ASGI lifecycle (e.g., adding middleware, mounting under a larger app).

### serve()

```python
def serve(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None: ...
```

Blocking convenience function. Calls `uvicorn.run()` internally. Best for scripts and local development.

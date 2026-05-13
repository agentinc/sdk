# RawAdapter — Detailed Reference

RawAdapter is the bridge between arbitrary Python functions and the AgentProtocol. It inspects function signatures at construction time and routes `AgentInput` fields to matching parameters at call time.

## Signature Detection Rules

RawAdapter checks parameters in this order:

### 1. First parameter is `message: str`

The function's first parameter must be named anything but annotated as `str`, or named `message` with no annotation. RawAdapter then checks for additional positional parameters:

**Just message:**
```python
async def agent(message: str) -> str:
    return f"Got: {message}"
```

**Message + history:**
```python
async def agent(message: str, history: list) -> str:
    prev = len(history)
    return f"Got: {message} (after {prev} messages)"
```

The `history` parameter receives a `list[dict]`, where each dict has:
```python
{"role": "user"|"assistant"|"tool", "content": "...", "tool_calls": [...], "tool_call_id": "..."}
```

**Message + history + tools:**
```python
async def agent(message: str, history: list, tools: list):
    # tools is a list of dicts: [{"name": "...", "description": "...", "parameters": {...}}]
    yield f"I have {len(tools)} tools"
```

The second parameter must be named `history` and the third must be named `tools` for auto-detection to work.

### 2. First parameter accepts AgentInput

If the first parameter is not a str-like, RawAdapter passes the full `AgentInput` object:

```python
from agentinc.sdk import AgentInput, AgentOutput

async def agent(input: AgentInput) -> AgentOutput:
    return AgentOutput(content=input.message, done=True)
```

## Return Type Handling

RawAdapter adapts the return value automatically:

### String return
```python
async def agent(message: str) -> str:
    return "hello"
# → yields AgentOutput(content="hello", done=True)
```

### AgentOutput return
```python
async def agent(input: AgentInput) -> AgentOutput:
    return AgentOutput(content="hello", tool_calls=[], done=True)
# → yields that AgentOutput as-is
```

### Async generator yielding strings (streaming)
```python
async def agent(message: str):
    yield "hello "
    yield "world"
# → yields AgentOutput(content="hello ", done=False)
# → yields AgentOutput(content="world", done=False)
# → yields AgentOutput(content="", done=True)  ← auto-appended sentinel
```

When an async generator yields only strings, RawAdapter appends a final `AgentOutput(content="", done=True)` after the generator exhausts.

### Async generator yielding dicts (tool calls)
```python
async def agent(message: str, history: list, tools: list):
    yield {
        "tool_calls": [
            {"id": "tc-1", "name": "search", "arguments": {"q": message}}
        ],
        "done": False,
    }
```

Dict chunks are converted: each entry in `tool_calls` becomes a `ToolCall` model, and the dict becomes an `AgentOutput`.

### Async generator yielding AgentOutput directly
```python
async def agent(input: AgentInput):
    yield AgentOutput(content="thinking...", done=False)
    yield AgentOutput(content="done!", done=True)
# → yields each AgentOutput as-is
```

### Mixed generators

You can mix strings and dicts in the same generator:
```python
async def agent(message: str, history: list, tools: list):
    yield "Let me search for that..."                # string → streaming text
    yield {"tool_calls": [...], "done": False}       # dict → tool call
    yield "Here's what I found: ..."                 # string → more text
```

### Sync functions

RawAdapter handles sync functions by awaiting the coroutine if needed:
```python
def agent(message: str) -> str:
    return "sync works too"
```

## Edge Cases

- If the function raises an exception, it propagates — RawAdapter does not catch errors.
- A function that returns `None` or a non-string/non-AgentOutput gets `str()` called on it and wrapped in `AgentOutput(content=str(result), done=True)`.
- Sync generators are not supported — use async generators.
- The parameter name detection is case-sensitive: `message`, `history`, `tools`.

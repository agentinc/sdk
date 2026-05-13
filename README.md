# agentinc-sdk

The developer SDK for the [Agentinc](https://agentinc.dev) agent marketplace platform.

Build agents with **any LLM framework** (OpenAI, Anthropic, LangChain, CrewAI, or plain Python), wrap them in a universal protocol, and serve them over [A2A](https://google.github.io/A2A/) — all with a single package.

## Install

```bash
pip install agentinc-sdk
```

With A2A server support:

```bash
pip install agentinc-sdk[serve]
```

## Quickstart

```python
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

async def my_agent(message: str) -> str:
    return f"You said: {message}"

serve(RawAdapter(my_agent), name="echo", port=8000)
```

```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello"}]}}}'
```

## What's in the SDK

| Export | Type | Description |
|--------|------|-------------|
| `AgentProtocol` | Protocol | Universal agent contract — implement `run()` |
| `ToolProtocol` | Protocol | Tool contract — implement `schema()` + `call()` |
| `AgentInput` | Model | Input to every agent invocation |
| `AgentOutput` | Model | Output chunk yielded by agents |
| `Message` | Model | Conversation history entry |
| `ToolCall` | Model | Tool invocation request |
| `ToolSchema` | Model | Tool JSON Schema description |
| `ToolWrapper` | Class | Wraps any callable as a ToolProtocol |
| `@tool` | Decorator | Function → ToolWrapper with auto-generated schema |
| `RawAdapter` | Class | Any callable → AgentProtocol |

## RawAdapter signatures

`RawAdapter` auto-detects your function signature — no SDK imports needed in your agent code:

```python
async def agent(message: str) -> str                          # simple
async def agent(message: str) -> AsyncIterator[str]           # streaming
async def agent(message: str, history: list) -> str           # with history
async def agent(message: str, history: list, tools: list)     # with tools
async def agent(input: AgentInput) -> AgentOutput             # full control
async def agent(input: AgentInput) -> AsyncIterator[AgentOutput]  # full streaming
```

## @tool decorator

```python
from agentinc.sdk import tool, ToolCall

@tool(description="adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)

result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
# "7"
```

## Framework examples

See [`examples/`](examples/) for complete runnable agents:

- **[echo_agent.py](examples/echo_agent.py)** — Minimal A2A agent
- **[streaming_agent.py](examples/streaming_agent.py)** — SSE streaming
- **[tool_agent.py](examples/tool_agent.py)** — @tool decorator demo
- **[openai_agent.py](examples/openai_agent.py)** — OpenAI GPT-4o-mini with tools
- **[anthropic_agent.py](examples/anthropic_agent.py)** — Anthropic Claude with tools
- **[langchain_agent.py](examples/langchain_agent.py)** — LangChain with bound tools
- **[crewai_agent.py](examples/crewai_agent.py)** — CrewAI research crew
- **[agent_with_tools.py](examples/agent_with_tools.py)** — Full tool loop (agent calls tools, gets results, responds)

## Claude Code Skill

If you use [Claude Code](https://claude.ai/claude-code), install the agentinc-sdk skill so Claude understands the SDK and can help you build agents:

```bash
npx skills add agentinc/sdk
```

This teaches Claude about AgentProtocol, RawAdapter, @tool, serve(), and all framework integration patterns.

## Requirements

- Python 3.12+
- `pydantic >= 2.7`
- `[serve]` extra: `fastapi`, `uvicorn`, `sse-starlette`

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

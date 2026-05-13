# Framework Integration Examples

Complete examples for wrapping agents from popular LLM frameworks with agentinc-sdk.

## Table of Contents

1. [OpenAI](#openai)
2. [Anthropic](#anthropic)
3. [LangChain](#langchain)
4. [CrewAI](#crewai)

---

## OpenAI

**Dependencies:** `pip install openai agentinc-sdk[serve]`

```python
import json
from openai import AsyncOpenAI
from agentinc.sdk import RawAdapter, tool
from agentinc.sdk.serve import serve

client = AsyncOpenAI()

@tool(description="gets the current weather for a city")
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

async def openai_agent(message: str, history: list, tools: list):
    # Build messages from history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # Convert agentinc tool schemas to OpenAI format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ] or None

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools,
    )

    choice = response.choices[0]

    if choice.message.tool_calls:
        yield {
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ],
            "done": False,
        }
    else:
        yield choice.message.content

if __name__ == "__main__":
    serve(RawAdapter(openai_agent), name="openai-agent", port=8000)
```

### Key points for OpenAI integration

- Convert `tools` list (agentinc format) to OpenAI's `{"type": "function", "function": {...}}` format
- Parse `tc.function.arguments` with `json.loads()` — OpenAI returns it as a JSON string
- Use `choice.message.tool_calls` to detect tool call responses vs text responses

---

## Anthropic

**Dependencies:** `pip install anthropic agentinc-sdk[serve]`

```python
from anthropic import AsyncAnthropic
from agentinc.sdk import RawAdapter, tool
from agentinc.sdk.serve import serve

client = AsyncAnthropic()

@tool(description="looks up a fact in the knowledge base")
def lookup(topic: str) -> str:
    return f"Here's what we know about {topic}: it's fascinating."

async def anthropic_agent(message: str, history: list, tools: list):
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # Convert agentinc tool schemas to Anthropic format
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],  # Anthropic uses input_schema, not parameters
        }
        for t in tools
    ] or None

    kwargs = dict(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a helpful assistant.",
        messages=messages,
    )
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    response = await client.messages.create(**kwargs)

    for block in response.content:
        if block.type == "tool_use":
            yield {
                "tool_calls": [
                    {"id": block.id, "name": block.name, "arguments": block.input}
                ],
                "done": False,
            }
        elif block.type == "text":
            yield block.text

if __name__ == "__main__":
    serve(RawAdapter(anthropic_agent), name="anthropic-agent", port=8002)
```

### Streaming variant

```python
async def anthropic_streaming(message: str):
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
```

### Key points for Anthropic integration

- Anthropic uses `input_schema` instead of `parameters` for tool definitions
- Anthropic uses `block.input` (already a dict) instead of a JSON string for tool arguments
- `system` is a top-level parameter, not a message
- Response content is a list of blocks — iterate and check `.type`

---

## LangChain

**Dependencies:** `pip install langchain-openai langchain-core agentinc-sdk[serve]`

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

@lc_tool
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([calculator])

async def langchain_agent(message: str, history: list):
    messages = [SystemMessage(content="You are a helpful assistant with a calculator.")]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))

    response = await llm.ainvoke(messages)

    if response.tool_calls:
        yield {
            "tool_calls": [
                {"id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
                for tc in response.tool_calls
            ],
            "done": False,
        }
    else:
        yield response.content

if __name__ == "__main__":
    serve(RawAdapter(langchain_agent), name="langchain-agent", port=8003)
```

### Streaming variant

```python
async def langchain_streaming(message: str):
    messages = [SystemMessage(content="You are helpful."), HumanMessage(content=message)]
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
```

### Key points for LangChain integration

- Use `bind_tools()` on the LLM, not on the agent
- LangChain tool calls use `tc["args"]` not `tc["arguments"]` — map accordingly
- Convert agentinc history dicts to LangChain message objects (`HumanMessage`, `AIMessage`)
- Use LangChain's own `@tool` decorator for LangChain tools, and agentinc's `@tool` for agentinc tools

---

## CrewAI

**Dependencies:** `pip install crewai agentinc-sdk[serve]`

```python
import asyncio
from crewai import Agent, Crew, Task
from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve

researcher = Agent(
    role="Senior Research Analyst",
    goal="Provide thorough, well-structured research on any topic",
    backstory="You are an experienced research analyst.",
    verbose=False,
    allow_delegation=False,
)

def run_crew(message: str) -> str:
    task = Task(
        description=message,
        expected_output="A clear, concise analysis",
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)

# CrewAI is synchronous — use asyncio.to_thread to avoid blocking
async def crewai_agent(message: str) -> str:
    return await asyncio.to_thread(run_crew, message)

if __name__ == "__main__":
    serve(RawAdapter(crewai_agent), name="crewai-agent", port=8004)
```

### Key points for CrewAI integration

- CrewAI's `Crew.kickoff()` is synchronous and blocking — always wrap with `asyncio.to_thread()`
- CrewAI manages its own LLM calls, so you don't convert tool schemas
- The agent function signature is simple `(message: str) -> str` since CrewAI handles orchestration internally
- Set `verbose=False` to avoid noisy logs in production

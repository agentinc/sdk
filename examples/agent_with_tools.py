"""
Agent that uses tools to answer questions.

Demonstrates the full tool loop: the agent decides which tool to call,
the runner dispatches the call, and the result is fed back to the agent.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/agent_with_tools.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the weather in Paris and what is 25 * 4?"}]}}}' | python -m json.tool
"""

import asyncio
import json

from openai import AsyncOpenAI

from agentinc.sdk import AgentInput, AgentOutput, RawAdapter, ToolCall, tool
from agentinc.sdk.serve import serve

client = AsyncOpenAI()


# ── Define tools ──

@tool(description="Gets the current weather for a city")
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"


@tool(description="Evaluates a math expression")
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool(description="Looks up a person's profile by name")
def lookup_person(name: str) -> str:
    profiles = {
        "alice": "Alice — Senior Engineer, joined 2021, works on infra",
        "bob": "Bob — Product Manager, joined 2023, owns the marketplace",
    }
    return profiles.get(name.lower(), f"No profile found for {name}")


TOOLS = [get_weather, calculator, lookup_person]
TOOL_MAP = {t.schema().name: t for t in TOOLS}


# ── Agent with tool loop ──

async def smart_agent(message: str, history: list, tools: list):
    messages = [{"role": "system", "content": "You are a helpful assistant. Use the tools available to answer questions accurately."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

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
    ]

    max_rounds = 5
    for _ in range(max_rounds):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            yield choice.message.content
            return

        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ],
        })

        for tc in choice.message.tool_calls:
            tool_fn = TOOL_MAP.get(tc.function.name)
            if tool_fn:
                result = await tool_fn.call(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
            else:
                result = f"Unknown tool: {tc.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    yield "Reached maximum tool rounds."


# ── Serve ──

if __name__ == "__main__":
    agent = RawAdapter(smart_agent)
    inp = AgentInput(
        message="startup",
        tool_schemas=[t.schema() for t in TOOLS],
    )

    serve(agent, name="smart-agent", description="Agent with weather, calculator, and lookup tools", port=8000)

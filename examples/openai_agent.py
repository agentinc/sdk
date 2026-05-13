"""
Agent using the OpenAI SDK, served over A2A.

Requires:
    pip install openai

Run:
    export OPENAI_API_KEY=sk-...
    python examples/openai_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the capital of France?"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Write a haiku about code"}]}}}'
"""

from openai import AsyncOpenAI

from agentinc.sdk import AgentInput, AgentOutput, RawAdapter, tool
from agentinc.sdk.serve import serve


client = AsyncOpenAI()


@tool(description="gets the current weather for a city")
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"


async def openai_agent(message: str, history: list, tools: list):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
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
                    "arguments": dict(__import__("json").loads(tc.function.arguments)),
                }
                for tc in choice.message.tool_calls
            ],
            "done": False,
        }
    else:
        yield choice.message.content


if __name__ == "__main__":
    serve(
        RawAdapter(openai_agent),
        name="openai-agent",
        description="Chat agent powered by OpenAI GPT-4o-mini with tool support",
        port=8000,
    )

"""
Agent using the Anthropic SDK, served over A2A.

Requires:
    pip install anthropic

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/anthropic_agent.py

Test:
    curl -s -X POST http://localhost:8002 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Explain quantum computing in 2 sentences"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8002 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Write a limerick about Python"}]}}}'
"""

from anthropic import AsyncAnthropic

from agentinc.sdk import AgentInput, AgentOutput, RawAdapter, tool
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

    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
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


async def anthropic_streaming_agent(message: str, history: list, tools: list):
    """Variant that streams tokens via Anthropic's streaming API."""
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a helpful assistant.",
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    serve(
        RawAdapter(anthropic_agent),
        name="anthropic-agent",
        description="Chat agent powered by Claude Sonnet with tool support",
        port=8002,
    )

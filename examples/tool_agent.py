"""
Agent with tools — demonstrates @tool decorator and ToolWrapper.

This example doesn't serve over HTTP; it runs the agent locally
to show how tools integrate with RawAdapter.

Run:
    python examples/tool_agent.py
"""

import asyncio

from agentinc.sdk import tool, RawAdapter, AgentInput, ToolCall


@tool(description="adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)


@tool(description="multiplies two numbers")
def multiply(a: float, b: float) -> str:
    return str(a * b)


async def calculator(message: str, history: list, tools: list) -> str:
    return f"I have {len(tools)} tools available. You said: {message}"


async def main():
    print("=== Tool schemas ===")
    print(f"  add: {add.schema().model_dump()}")
    print(f"  multiply: {multiply.schema().model_dump()}")

    print("\n=== Direct tool calls ===")
    result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
    print(f"  add(3, 4) = {result}")

    result = await multiply.call(ToolCall(id="2", name="multiply", arguments={"a": 5, "b": 6}))
    print(f"  multiply(5, 6) = {result}")

    print("\n=== Agent with tool awareness ===")
    agent = RawAdapter(calculator)
    inp = AgentInput(
        message="what is 3+4?",
        tool_schemas=[add.schema(), multiply.schema()],
    )
    async for output in agent.run(inp):
        print(f"  agent: {output.content}")


if __name__ == "__main__":
    asyncio.run(main())

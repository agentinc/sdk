"""
Agent using LangChain, served over A2A.

Requires:
    pip install langchain-openai langchain-core

Run:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_agent.py

Test:
    curl -s -X POST http://localhost:8003 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is 25 * 4?"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8003 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Tell me a joke"}]}}}'
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from agentinc.sdk import AgentInput, AgentOutput, RawAdapter
from agentinc.sdk.serve import serve


@lc_tool
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        return str(eval(expression))  # noqa: S307 — example only
    except Exception as e:
        return f"Error: {e}"


llm = ChatOpenAI(model="gpt-4o-mini", streaming=True).bind_tools([calculator])


async def langchain_agent(message: str, history: list):
    messages = [SystemMessage(content="You are a helpful assistant with a calculator tool.")]
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
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["args"],
                }
                for tc in response.tool_calls
            ],
            "done": False,
        }
    else:
        yield response.content


async def langchain_streaming_agent(message: str):
    """Variant that streams tokens."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message),
    ]
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    serve(
        RawAdapter(langchain_agent),
        name="langchain-agent",
        description="LangChain agent with calculator tool",
        port=8003,
    )

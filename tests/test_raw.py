import pytest
from agentinc.sdk import RawAdapter, AgentInput, AgentOutput


@pytest.mark.asyncio
async def test_raw_sync_str():
    def echo(message: str) -> str:
        return f"echo: {message}"

    adapter = RawAdapter(echo)
    outputs = [o async for o in adapter.run(AgentInput(message="hi"))]
    assert len(outputs) == 1
    assert outputs[0].content == "echo: hi"
    assert outputs[0].done is True


@pytest.mark.asyncio
async def test_raw_async_str():
    async def echo(message: str) -> str:
        return f"echo: {message}"

    adapter = RawAdapter(echo)
    outputs = [o async for o in adapter.run(AgentInput(message="hi"))]
    assert len(outputs) == 1
    assert outputs[0].content == "echo: hi"


@pytest.mark.asyncio
async def test_raw_async_gen_str():
    async def stream(message: str):
        yield "hello "
        yield "world"

    adapter = RawAdapter(stream)
    outputs = [o async for o in adapter.run(AgentInput(message="hi"))]
    assert outputs[0].content == "hello "
    assert outputs[1].content == "world"
    assert outputs[-1].done is True


@pytest.mark.asyncio
async def test_raw_agent_input():
    async def agent(input: AgentInput) -> AgentOutput:
        return AgentOutput(content=f"got: {input.message}", done=True)

    adapter = RawAdapter(agent)
    outputs = [o async for o in adapter.run(AgentInput(message="test"))]
    assert outputs[0].content == "got: test"


@pytest.mark.asyncio
async def test_raw_with_history():
    async def agent(message: str, history: list) -> str:
        return f"{message} (history={len(history)})"

    adapter = RawAdapter(agent)
    outputs = [o async for o in adapter.run(AgentInput(message="hi", history=[]))]
    assert "history=0" in outputs[0].content

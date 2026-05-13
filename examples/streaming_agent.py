"""
Streaming agent that yields chunks over A2A SSE.

Run:
    python examples/streaming_agent.py

Test:
    curl -N -X POST http://localhost:8001 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"tell me a story"}]}}}'
"""

import asyncio

from agentinc.sdk import AgentInput, AgentOutput, RawAdapter
from agentinc.sdk.serve import serve


async def storyteller(message: str):
    words = f"Once upon a time, someone said: {message}. The end.".split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    serve(RawAdapter(storyteller), name="storyteller", description="Streams a short story", port=8001)

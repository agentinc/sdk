"""
Minimal echo agent served over A2A.

Run:
    python examples/echo_agent.py

Test (in another terminal):
    # Single request
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello world"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello world"}]}}}'

    # Agent card
    curl -s http://localhost:8000/.well-known/agent.json | python -m json.tool
"""

from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve


async def echo(message: str) -> str:
    return f"Echo: {message}"


if __name__ == "__main__":
    serve(RawAdapter(echo), name="echo-agent", description="Echoes your message back", port=8000)

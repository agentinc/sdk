"""
Agent using CrewAI, served over A2A.

Requires:
    pip install crewai crewai-tools

Run:
    export OPENAI_API_KEY=sk-...
    python examples/crewai_agent.py

Test:
    curl -s -X POST http://localhost:8004 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Research the benefits of TypeScript over JavaScript"}]}}}' | python -m json.tool
"""

import asyncio

from crewai import Agent, Crew, Task

from agentinc.sdk import RawAdapter
from agentinc.sdk.serve import serve


researcher = Agent(
    role="Senior Research Analyst",
    goal="Provide thorough, well-structured research on any topic",
    backstory="You are an experienced research analyst who excels at breaking down complex topics.",
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


async def crewai_agent(message: str) -> str:
    return await asyncio.to_thread(run_crew, message)


if __name__ == "__main__":
    serve(
        RawAdapter(crewai_agent),
        name="crewai-agent",
        description="CrewAI research analyst agent",
        port=8004,
    )

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

from app.config import (
    OPENAI_API_KEY,
    SERPER_API_KEY,
    LLM_MODEL,
)


def create_web_resolution_agent():
    if not SERPER_API_KEY:
        raise ValueError(
            "SERPER_API_KEY is missing from .env"
        )

    llm = LLM(
        model=f"openai/{LLM_MODEL}",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )

    search_tool = SerperDevTool(
        api_key=SERPER_API_KEY
    )

    return Agent(
        role="Customer Support Web Resolution Specialist",
        goal=(
            "Research reliable web information only when internal "
            "support documentation is insufficient."
        ),
        backstory=(
            "You are a senior support specialist who uses web research "
            "only as a fallback. You prefer authoritative sources and "
            "avoid unsupported assumptions."
        ),
        llm=llm,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False,
    )


def resolve_using_web(
    query: str,
    rag_answer: str,
    confidence: int,
):
    print("=" * 70)
    print("AGENT 3 - WEB SEARCH RESOLUTION")
    print("=" * 70)

    agent = create_web_resolution_agent()

    task = Task(
        description=f"""
CUSTOMER QUERY:
{query}

INTERNAL KNOWLEDGE-BASE RESULT:
{rag_answer}

VALIDATION CONFIDENCE:
{confidence}%

The internal knowledge base was not sufficient.

INSTRUCTIONS:

1. Search the web for reliable information relevant to the query.
2. Prefer official, authoritative, or primary sources.
3. Do not invent facts.
4. Do not claim company-specific policies unless you find reliable
   evidence for them.
5. If reliable information cannot be found, clearly say that the
   available information is insufficient.
6. Give a concise and customer-friendly response.
7. Include useful steps when supported by reliable sources.
8. Do not mention RAG, FAISS, confidence scores, CrewAI,
   validation agents, or internal architecture.
""",
        expected_output=(
            "A concise customer-support response based on reliable "
            "web research."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    return str(result)


def run_resolution_agent(
    query: str,
    rag_answer: str,
    confidence: int,
    decision: str,
):
    if decision == "RAG":
        print("=" * 70)
        print("AGENT 3 - RAG ROUTE")
        print("=" * 70)

        # Important:
        # Do NOT call another LLM.
        # Agent 1 already produced the verified customer-ready answer.
        return {
            "route": "RAG",
            "final_answer": rag_answer,
        }

    final_answer = resolve_using_web(
        query=query,
        rag_answer=rag_answer,
        confidence=confidence,
    )

    return {
        "route": "WEB_SEARCH",
        "final_answer": final_answer,
    }
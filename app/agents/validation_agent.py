from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM

from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    CONFIDENCE_THRESHOLD,
)


class ValidationResult(BaseModel):
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score from 0 to 100"
    )

    decision: str = Field(
        description="Either RAG or WEB_SEARCH"
    )

    reason: str = Field(
        description="Short explanation for the decision"
    )


def create_validation_agent():
    llm = LLM(
        model=f"openai/{LLM_MODEL}",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    validation_agent = Agent(
        role="Customer Support Response Validator",
        goal=(
            "Evaluate whether the RAG answer is sufficiently supported "
            "by the retrieved knowledge-base evidence."
        ),
        backstory=(
            "You are a quality-control specialist for a customer support "
            "AI system. You carefully verify whether answers are grounded "
            "in internal documentation. You do not reward answers merely "
            "because they sound plausible."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return validation_agent

def run_validation_agent(
    query: str,
    rag_answer: str,
    retrieved_items: list,
):
    print("=" * 70)
    print("AGENT 2 - VALIDATION AGENT")
    print("=" * 70)

    evidence_parts = []

    for index, item in enumerate(
        retrieved_items,
        start=1,
    ):
        evidence_parts.append(
            f"""
SOURCE {index}

Page:
{item.get("page")}

FAISS Distance:
{item.get("distance")}

Content:
{item.get("content")}
"""
        )

    evidence = "\n".join(evidence_parts)

    validation_agent = create_validation_agent()

    task = Task(
        description=f"""
Evaluate the following customer-support RAG response.

CUSTOMER QUERY:
{query}

RAG ANSWER:
{rag_answer}

RETRIEVED KNOWLEDGE-BASE EVIDENCE:

{evidence}

You must determine how strongly the answer is supported by
the retrieved internal documentation.

CONFIDENCE RULES:

90-100:
The retrieved knowledge base directly and clearly answers
the customer's question.

75-89:
The evidence strongly supports the answer, but some small
details may be indirect.

71-74:
The evidence is relevant and probably sufficient, but is
not completely direct.

40-70:
Some relevant information exists, but it is incomplete,
unclear, or insufficient.

0-39:
The knowledge base does not meaningfully answer the
customer's question.

Decision rule:

If confidence is greater than {CONFIDENCE_THRESHOLD},
decision must be:

RAG

If confidence is less than or equal to
{CONFIDENCE_THRESHOLD},
decision must be:

WEB_SEARCH

Important rules:

1. Do not invent evidence.
2. Do not judge based only on how good the answer sounds.
3. Validate the answer against the retrieved evidence.
4. FAISS distance is retrieval information only.
5. Do not convert FAISS distance directly into a percentage.
6. Be conservative when information is incomplete.
""",
        expected_output=(
            "A structured validation result containing confidence, "
            "decision, and reason."
        ),
        agent=validation_agent,
        output_pydantic=ValidationResult,
    )

    crew = Crew(
        agents=[validation_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    validation_output = result.pydantic

    if validation_output.confidence > CONFIDENCE_THRESHOLD:
        validation_output.decision = "RAG"
    else:
        validation_output.decision = "WEB_SEARCH"

    return validation_output

if __name__ == "__main__":

    from app.agents.rag_agent import run_rag_agent

    customer_query = input(
        "Enter customer query: "
    )

    rag_result = run_rag_agent(
        customer_query
    )

    validation_result = run_validation_agent(
        query=customer_query,
        rag_answer=rag_result["answer"],
        retrieved_items=rag_result["retrieved_items"],
    )
    
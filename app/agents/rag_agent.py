from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
)


def load_vector_store():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_context(query: str, k: int = 4):
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    items = []

    for document, score in results:
        items.append(
            {
                "content": document.page_content,
                "page": document.metadata.get("page"),
                "source": document.metadata.get("source"),
                "distance": float(score),
            }
        )

    return items


def build_context(retrieved_items):
    parts = []

    for index, item in enumerate(retrieved_items, start=1):
        parts.append(
            f"""
SOURCE {index}
Page: {item['page']}
FAISS Distance: {item['distance']}

Content:
{item['content']}
"""
        )

    return "\n".join(parts)


def create_rag_agent():
    llm = LLM(
        model=f"openai/{LLM_MODEL}",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    return Agent(
        role="Customer Support RAG Specialist",
        goal=(
            "Answer customer support questions accurately using only "
            "the retrieved internal customer-support documentation."
        ),
        backstory=(
            "You are a careful customer-support specialist. "
            "You never invent procedures, URLs, policies, timelines, "
            "or troubleshooting steps."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def run_rag_agent(query: str):
    print("=" * 70)
    print("AGENT 1 - RAG AGENT")
    print("=" * 70)

    retrieved_items = retrieve_context(
        query=query,
        k=4,
    )

    context = build_context(retrieved_items)

    agent = create_rag_agent()

    task = Task(
        description=f"""
CUSTOMER QUERY:
{query}

RETRIEVED INTERNAL KNOWLEDGE BASE:
{context}

INSTRUCTIONS:

1. Answer the customer's question directly.
2. Use ONLY the retrieved knowledge-base content.
3. Preserve exact procedures, limits, conditions, and timelines.
4. If steps are available, present them clearly as numbered steps.
5. Do not replace specific documentation with generic advice.
6. Do not invent URLs, policies, products, features, or procedures.
7. Do not perform a web search.
8. If the knowledge base does not contain enough information,
   clearly say:
   "The internal knowledge base does not contain enough information
   to answer this question."
9. Write the response so it can be sent directly to the customer.
10. Do not mention FAISS, embeddings, retrieval scores, CrewAI,
    agents, or internal AI architecture.
""",
        expected_output=(
            "A concise, accurate, customer-ready support response "
            "grounded only in the provided documentation."
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

    return {
        "query": query,
        "answer": str(result),
        "retrieved_items": retrieved_items,
    }
import time

from app.agents.rag_agent import (
    run_rag_agent,
)

from app.agents.validation_agent import (
    run_validation_agent,
)

from app.agents.resolution_agent import (
    run_resolution_agent,
)

from app.services.logger import (
    save_conversation_log,
)


def run_customer_support(
    username: str,
    query: str,
):
    start_time = time.time()

    print("=" * 70)
    print("CUSTOMER SUPPORT CREW STARTED")
    print("=" * 70)

    # -------------------------------------------------
    # AGENT 1
    # -------------------------------------------------

    print("\nRunning Agent 1 - RAG")

    rag_result = run_rag_agent(
        query
    )

    # -------------------------------------------------
    # AGENT 2
    # -------------------------------------------------

    print("\nRunning Agent 2 - Validation")

    validation_result = run_validation_agent(
        query=query,
        rag_answer=rag_result["answer"],
        retrieved_items=rag_result[
            "retrieved_items"
        ],
    )

    # -------------------------------------------------
    # AGENT 3
    # -------------------------------------------------

    print("\nRunning Agent 3 - Resolution")

    resolution_result = run_resolution_agent(
        query=query,
        rag_answer=rag_result["answer"],
        confidence=validation_result.confidence,
        decision=validation_result.decision,
    )

    # -------------------------------------------------
    # LOGGING
    # -------------------------------------------------

    log_result = save_conversation_log(
        username=username,
        query=query,
        rag_answer=rag_result["answer"],
        confidence=validation_result.confidence,
        validation_reason=validation_result.reason,
        decision=validation_result.decision,
        final_answer=resolution_result[
            "final_answer"
        ],
        start_time=start_time,
    )

    # -------------------------------------------------
    # FINAL RESPONSE
    # -------------------------------------------------

    return {
        "username": username,
        "query": query,

        "rag_answer":
            rag_result["answer"],

        "confidence":
            validation_result.confidence,

        "validation_reason":
            validation_result.reason,

        "decision":
            validation_result.decision,

        "route":
            resolution_result["route"],

        "final_answer":
            resolution_result[
                "final_answer"
            ],

        "response_time":
            log_result["response_time"],

        "log_file":
            log_result["filepath"],
    }

if __name__ == "__main__":

    username = input(
        "Enter customer email: "
    )

    query = input(
        "Enter customer query: "
    )

    result = run_customer_support(
        username=username,
        query=query,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Customer: "
        f"{result['username']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}%"
    )

    print(
        f"Route: "
        f"{result['route']}"
    )

    print(
        f"Response Time: "
        f"{result['response_time']} seconds"
    )

    print("\nFinal Answer:\n")

    print(
        result["final_answer"]
    )

    print(
        f"\nLog file: "
        f"{result['log_file']}"
    )
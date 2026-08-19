import os
import time
from datetime import datetime

from app.config import CONVERSATION_LOG_PATH


def save_conversation_log(
    username: str,
    query: str,
    rag_answer: str,
    confidence: int,
    validation_reason: str,
    decision: str,
    final_answer: str,
    start_time: float,
):
    os.makedirs(
        CONVERSATION_LOG_PATH,
        exist_ok=True,
    )

    now = datetime.now()

    total_time = round(
        time.time() - start_time,
        2,
    )

    safe_username = (
        username.replace("@", "_at_")
        .replace(".", "_")
        .replace(" ", "_")
    )

    filename = (
        f"{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        f"_{safe_username}.txt"
    )

    filepath = os.path.join(
        CONVERSATION_LOG_PATH,
        filename,
    )

    log_content = f"""
CUSTOMER SUPPORT CONVERSATION
============================================================

Username:
{username}

Date:
{now.strftime('%Y-%m-%d')}

Time:
{now.strftime('%H:%M:%S')}

------------------------------------------------------------
CUSTOMER QUERY
------------------------------------------------------------

{query}

------------------------------------------------------------
AGENT 1 - RAG RESPONSE
------------------------------------------------------------

{rag_answer}

------------------------------------------------------------
AGENT 2 - VALIDATION
------------------------------------------------------------

Confidence:
{confidence}%

Decision:
{decision}

Reason:
{validation_reason}

------------------------------------------------------------
AGENT 3 - FINAL RESPONSE
------------------------------------------------------------

{final_answer}

------------------------------------------------------------
EXECUTION DETAILS
------------------------------------------------------------

Route:
{decision}

Total Response Time:
{total_time} seconds

============================================================
END OF CONVERSATION
============================================================
"""

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(log_content)

    return {
        "filepath": filepath,
        "response_time": total_time,
    }
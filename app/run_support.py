import sys
import json
import base64
import contextlib
import io

from app.crew import run_customer_support


def decode_base64(value: str) -> str:
    return base64.b64decode(
        value.encode("utf-8")
    ).decode("utf-8")


def main():
    if len(sys.argv) < 3:
        print(
            json.dumps({
                "success": False,
                "error": "username and query are required"
            })
        )
        sys.exit(1)

    try:
        username = decode_base64(sys.argv[1])
        query = decode_base64(sys.argv[2])

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            result = run_customer_support(
                username=username,
                query=query,
            )

        output = {
            "success": True,
            "username": result["username"],
            "query": result["query"],
            "rag_answer": result["rag_answer"],
            "confidence": result["confidence"],
            "validation_reason": result["validation_reason"],
            "decision": result["decision"],
            "route": result["route"],
            "final_answer": result["final_answer"],
            "response_time": result["response_time"],
            "log_file": result["log_file"],
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False
            )
        )

    except Exception as exc:
        print(
            json.dumps({
                "success": False,
                "error": str(exc),
            })
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

from app.crew import run_customer_support

#python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

app = FastAPI(
    title="Customer Support CrewAI API",
    description="FastAPI backend for CrewAI + FAISS customer support system",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    username: EmailStr
    query: str


class SupportResponse(BaseModel):
    username: str
    query: str
    rag_answer: str
    confidence: int
    validation_reason: str
    decision: str
    route: str
    final_answer: str
    response_time: float
    log_file: str


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Customer Support CrewAI API",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post(
    "/support/query",
    response_model=SupportResponse,
)
def support_query(
    request: SupportRequest,
):
    try:

        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        result = run_customer_support(
            username=request.username,
            query=request.query.strip(),
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"Support API error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
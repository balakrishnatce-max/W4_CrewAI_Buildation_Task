import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "70")
)

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "data/faiss/customer_support"
)

CONVERSATION_LOG_PATH = os.getenv(
    "CONVERSATION_LOG_PATH",
    "data/conversations"
)

PDF_PATH = os.getenv(
    "PDF_PATH",
    "data/pdf/customer_support.pdf"
)


def validate_config():
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file."
        )

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Support PDF not found: {PDF_PATH}"
        )

    return True
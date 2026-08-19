from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CONFIDENCE_THRESHOLD,
    FAISS_INDEX_PATH,
    PDF_PATH,
    validate_config,
)


def main():
    print("=" * 60)
    print("CUSTOMER SUPPORT AI - ENVIRONMENT TEST")
    print("=" * 60)

    print(f"API key loaded: {'YES' if OPENAI_API_KEY else 'NO'}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"PDF path: {PDF_PATH}")
    print(f"FAISS path: {FAISS_INDEX_PATH}")

    validate_config()

    print("\nEnvironment configuration: OK")
    print("Support PDF: FOUND")


if __name__ == "__main__":
    main()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def main():
    print("=" * 60)
    print("FAISS RETRIEVAL TEST")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    query = "How do I reset my password?"

    results = vector_store.similarity_search_with_score(
        query,
        k=3,
    )

    print(f"\nQuery: {query}")
    print("\nTop results:")

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Result {index}")
        print(f"FAISS distance: {score}")
        print(f"Page: {document.metadata.get('page')}")
        print()
        print(document.page_content[:700])


if __name__ == "__main__":
    main()
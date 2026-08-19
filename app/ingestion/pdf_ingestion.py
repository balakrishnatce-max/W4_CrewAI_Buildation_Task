import os

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    PDF_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def load_pdf(pdf_path: str):
    print(f"Reading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_number,
                },
            )
        )

    print(f"Pages loaded: {len(documents)}")

    return documents


def split_documents(documents):
    print("Splitting PDF content into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    return chunks


def create_embeddings():
    print(f"Using embedding model: {EMBEDDING_MODEL}")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    return embeddings


def create_faiss_index(chunks):
    print("Creating FAISS vector database...")

    embeddings = create_embeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_faiss_index(vector_store):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    vector_store.save_local(
        FAISS_INDEX_PATH
    )

    print(
        f"FAISS index saved successfully to: "
        f"{FAISS_INDEX_PATH}"
    )


def build_vector_database():
    print("=" * 60)
    print("CUSTOMER SUPPORT FAISS INGESTION")
    print("=" * 60)

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    documents = load_pdf(PDF_PATH)

    if not documents:
        raise ValueError(
            "No readable text found inside the PDF."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    vector_store = create_faiss_index(chunks)

    save_faiss_index(vector_store)

    print("\n" + "=" * 60)
    print("FAISS KNOWLEDGE BASE CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    build_vector_database()
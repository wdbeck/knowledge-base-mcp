import os
import time
import uuid
from pathlib import Path
from typing import List

from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone


# ==========================================
# CONFIG
# ==========================================

DOCS_DIR = Path("docs")  # <-- The folder you requested

# Chunk sizes
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 200

# Pinecone
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")
PINECONE_INDEX_HOST = "https://discovered-knowledge-base-2sn742y.svc.aped-4627-b74a.pinecone.io"
#os.environ.get("PINECONE_INDEX_HOST")

# Embeddings
EMBED_MODEL = "text-embedding-3-large"
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 64

KB_LABEL = "manuals"  # Label for all docs added by this script


# ==========================================
# CLIENTS
# ==========================================

client = OpenAI()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=PINECONE_INDEX_HOST)


# ==========================================
# Helpers
# ==========================================

def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text_parts.append(text)
    return "\n".join(text_parts)


def read_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""  # skip unsupported formats


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join([l for l in lines if l])


def chunk_text(text: str) -> List[str]:
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if not para.strip():
            continue

        if len(current) + len(para) + 1 > MAX_CHUNK_CHARS:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current.strip())
            current = para
        else:
            current = para if not current else current + "\n" + para

    if current and len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current.strip())

    if not chunks and text.strip():
        chunks = [text[:MAX_CHUNK_CHARS]]

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        print(f"Embedding {i}–{i+len(batch)}")
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        for item in resp.data:
            vectors.append(item.embedding)
        time.sleep(0.2)
    return vectors


def upsert_to_pinecone(records: List[dict]):
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[i:i+UPSERT_BATCH_SIZE]
        print(f"Upserting {len(batch)} records ({i}–{i+len(batch)})")

        kwargs = {}
        if PINECONE_NAMESPACE:
            kwargs["namespace"] = PINECONE_NAMESPACE

        index.upsert(
            vectors=[(r["id"], r["vector"], r["metadata"]) for r in batch],
            **kwargs
        )


# ==========================================
# MAIN
# ==========================================

def main():
    if not DOCS_DIR.exists():
        raise RuntimeError(f"docs folder not found at: {DOCS_DIR}")

    all_chunks = []
    all_metadata = []

    print(f"Scanning folder: {DOCS_DIR}")

    for path in DOCS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".pdf", ".txt", ".md"]:
            print(f"Reading: {path.name}")

            text = read_file_text(path)
            text = normalize_text(text)
            chunks = chunk_text(text)

            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "kb": KB_LABEL,
                    "source_file": str(path.relative_to(DOCS_DIR)),
                    "chunk_index": c_idx,
                    "text": chunk
                })

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("Nothing to embed. Exiting.")
        return

    print("Embedding...")
    vectors = embed_texts(all_chunks)

    print("Preparing records...")
    records = [
        {
            "id": str(uuid.uuid4()),
            "vector": vec,
            "metadata": meta
        }
        for vec, meta in zip(vectors, all_metadata)
    ]

    print("Upserting...")
    upsert_to_pinecone(records)

    print("✅ DONE — documents added to Pinecone.")


if __name__ == "__main__":
    main()
import os
import time
import uuid
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import tldextract

from openai import OpenAI
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

# =========================
# CONFIG
# =========================

# Your ProductFruits KB root URL (adjust if needed)
BASE_URL = "https://discoveredats.productfruits.help/en"

# How many pages max to crawl (safety limit)
MAX_PAGES = 300

# Chunking settings (simple heuristic)
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 200

# Pinecone settings
PINECONE_NAMESPACE = ""
PINECONE_INDEX_HOST = "https://discovered-knowledge-base-2sn742y.svc.aped-4627-b74a.pinecone.io"  # e.g. "your-index-xxxx.svc.pinecone.io"

# Batch sizes
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 64

# Models
EMBEDDING_MODEL = "text-embedding-3-large"  # or text-embedding-3-small for cheaper :contentReference[oaicite:0]{index=0}

# =========================
# CLIENTS
# =========================

client = OpenAI()  # reads OPENAI_API_KEY from env :contentReference[oaicite:1]{index=1}
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=PINECONE_INDEX_HOST)  # Pinecone v5+ style :contentReference[oaicite:2]{index=2}


# =========================
# CRAWLER
# =========================

def same_site(url: str, base_url: str) -> bool:
    """Limit crawling to the same registered domain (and scheme)."""
    base_ext = tldextract.extract(base_url)
    url_ext = tldextract.extract(url)

    same_domain = (base_ext.registered_domain == url_ext.registered_domain)
    same_scheme = urlparse(url).scheme in ("http", "https")
    return same_domain and same_scheme


def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes for consistency."""
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="", query="").geturl()
    # Remove trailing slash except for root
    if normalized.endswith("/") and len(normalized) > len(f"{parsed.scheme}://{parsed.netloc}/"):
        normalized = normalized[:-1]
    return normalized


def crawl_site(start_url: str, max_pages: int = 200):
    """Breadth-first crawl, returning {url: html}."""
    visited = set()
    queue = deque([start_url])
    pages = {}

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        url = normalize_url(url)

        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"Crawling: {url}")
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        html = resp.text
        pages[url] = html

        # Parse links
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Convert relative -> absolute
            next_url = urljoin(url, href)
            next_url = normalize_url(next_url)
            if same_site(next_url, start_url) and next_url not in visited:
                queue.append(next_url)

    print(f"Total pages crawled: {len(pages)}")
    return pages


# =========================
# TEXT EXTRACTION & CHUNKING
# =========================

def extract_main_text(html: str) -> str:
    """Very simple main-text extractor; you can customize for ProductFruits."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles/nav, etc.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # If ProductFruits has a specific content container, target it here:
    # main = soup.select_one(".pf-article-content") or soup.body
    main = soup.body or soup

    if not main:
        return ""

    text = main.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS):
    """Split text into overlapping-ish chunks by paragraphs."""
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if not para.strip():
            continue

        # If adding this paragraph would exceed max_chars, flush current
        if len(current) + len(para) + 1 > max_chars:
            if len(current) >= min_chars:
                chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n" + para
            else:
                current = para

    if current and len(current) >= min_chars:
        chunks.append(current.strip())

    # If nothing met min_chars, just return the whole thing as one chunk
    if not chunks and text.strip():
        chunks = [text[:max_chars]]

    return chunks


# =========================
# EMBEDDINGS
# =========================

def embed_texts(texts):
    """Return list of embedding vectors for given list of strings."""
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        print(f"Embedding batch {i}–{i + len(batch)}")
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )  # :contentReference[oaicite:3]{index=3}
        for item in resp.data:
            embeddings.append(item.embedding)
        # Tiny sleep to be polite to the API (tune as needed)
        time.sleep(0.2)
    return embeddings


# =========================
# PINECONE UPSERT
# =========================

def upsert_chunks_to_pinecone(records):
    """
    records is a list of dicts:
      {
        "id": str,
        "vector": list[float],
        "metadata": dict
      }
    """
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[i:i + UPSERT_BATCH_SIZE]
        print(f"Upserting {len(batch)} records to Pinecone: {i}–{i + len(batch)}")
        kwargs = {}
        if PINECONE_NAMESPACE:
            kwargs["namespace"] = PINECONE_NAMESPACE

        index.upsert(
            vectors=[
                (rec["id"], rec["vector"], rec["metadata"])
                for rec in batch
            ],
            **kwargs,
        )  # :contentReference[oaicite:4]{index=4}


# =========================
# MAIN REINDEX LOGIC
# =========================

def main():
    # Basic sanity checks
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var not set")
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY env var not set")
    if not PINECONE_INDEX_HOST:
        raise RuntimeError("PINECONE_INDEX_HOST env var not set")

    print("Starting crawl...")
    pages = crawl_site(BASE_URL, max_pages=MAX_PAGES)

    all_chunks = []
    chunk_metadatas = []

    print("Extracting and chunking...")
    for url, html in pages.items():
        text = extract_main_text(html)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadatas.append({
                "source_url": url,
                "chunk_index": idx,
                "kb": "productfruits",
            })

    print(f"Total text chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No content found, aborting.")
        return

    print("Generating embeddings...")
    vectors = embed_texts(all_chunks)

    print("Preparing records...")
    records = []
    for text, emb, meta in zip(all_chunks, vectors, chunk_metadatas):
        rec_id = str(uuid.uuid4())
        meta = dict(meta)
        meta["text"] = text  # store raw text for debugging/search
        records.append({
            "id": rec_id,
            "vector": emb,
            "metadata": meta,
        })

    print(f"Clearing namespace '{PINECONE_NAMESPACE or '__default__'}' before reindex...")
    try:
        if PINECONE_NAMESPACE:
            # Custom namespace
            index.delete(namespace=PINECONE_NAMESPACE, delete_all=True)
        else:
            # Default namespace – omit namespace argument
            index.delete(delete_all=True)
        print("Namespace cleared.")
    except NotFoundException:
        print(f"Namespace '{PINECONE_NAMESPACE or '__default__'}' does not exist yet; skipping clear.")

    print("Upserting into Pinecone...")
    upsert_chunks_to_pinecone(records)

    print("✅ Reindexing complete.")


if __name__ == "__main__":
    main()
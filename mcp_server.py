"""
MCP Server for Discovered Knowledge Base (Pinecone + OpenAI)

This server implements the Model Context Protocol (MCP) and exposes tools to:
- search: semantic search over a Pinecone index using OpenAI embeddings
- fetch: retrieve full chunk metadata/text by id from Pinecone
"""

import logging
import os
from typing import Dict, List, Any

from fastmcp import FastMCP
from openai import OpenAI
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Environment / config ===

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# For serverless Pinecone, this should be the index *host*,
# e.g. "your-index-xxxx.svc.pinecone.io"
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")

# Namespace: "" means default namespace (same as __default__ in UI)
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")  # 3072-dim
INDEX_DIM = int(os.environ.get("INDEX_DIM", "3072"))  # for info/logging only

# === Clients ===

openai_client = OpenAI() if OPENAI_API_KEY else None
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(host=PINECONE_INDEX_HOST) if pc and PINECONE_INDEX_HOST else None

server_instructions = """
This MCP server provides semantic search and retrieval over Discovered's
knowledge base and internal manuals (hiring best practices, assessment manual, etc.).

Tools:
- search(query, top_k, kb): find relevant chunks using OpenAI embeddings + Pinecone.
- fetch(id): retrieve full chunk text/metadata by chunk id from Pinecone.

When you use these tools to help a Discovered client:

1. Write an answer that explains, in simple, step-by-step terms,
   how the client can do what they are trying to do in Discovered.
2. Use the search results as your source of truth.
3. At the end of your answer, list the most relevant knowledge base article(s)
   (usually 1, max 3) with their titles and links (from `source_url` or similar metadata),
   so the client can read more if they want.
4. If the knowledge base doesn’t cover the request, say so explicitly and offer
   the closest guidance you can, but do not make up product behaviors.
"""


def create_server():
    """Create and configure the MCP server with search and fetch tools."""

    if not openai_client:
        raise ValueError("OPENAI_API_KEY is required")
    if not index:
        raise ValueError(
            "Pinecone index is not initialized. "
            "Make sure PINECONE_API_KEY and PINECONE_INDEX_HOST are set."
        )

    mcp = FastMCP(
        name="Discovered Knowledge Base MCP",
        instructions=server_instructions,
        stateless_http=True,
    )

    @mcp.tool()
    async def search(query: str, top_k: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for relevant documentation and manuals.

        Always use this tool before answering a Discovered product question.

        The caller (LLM) should:
        - Read the returned snippets and metadata.
        - Write a simple, step-by-step explanation for the user.
        - Then list 1–3 related KB articles using `source_url` / `source_file`
        so the user can click through for more detail.

        Args:
            query: Natural language question or topic.
            top_k: Number of results to return.
            kb: Optional filter (e.g. "productfruits", "manuals").

        Returns:
            {"results": [{ "id", "score", "text", "source_url", "metadata" }, ...]}
        """
        query = (query or "").strip()
        if not query:
            return {"results": []}

        logger.info(f"Embedding query with model '{EMBED_MODEL}': {query!r}")
        emb_resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=query,
        )
        query_vec = emb_resp.data[0].embedding

        kwargs = {"top_k": top_k, "vector": query_vec, "include_metadata": True}
        if PINECONE_NAMESPACE:
            kwargs["namespace"] = PINECONE_NAMESPACE

        logger.info(
            f"Querying Pinecone index at host '{PINECONE_INDEX_HOST}' "
            f"namespace='{PINECONE_NAMESPACE or '__default__'}', top_k={top_k}"
        )
        resp = index.query(**kwargs)

        results: List[Dict[str, Any]] = []

        for match in resp.matches or []:
            meta = match.metadata or {}
            text = meta.get("text", "")
            source_url = meta.get("source_url")

            # small snippet for display
            snippet = text[:200] + "..." if len(text) > 200 else text

            results.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": snippet,
                    "source_url": source_url,
                    "metadata": meta,
                }
            )

        logger.info(f"Pinecone search returned {len(results)} results")
        return {"results": results}

    @mcp.tool()
    async def fetch(id: str) -> Dict[str, Any]:
        """
        Retrieve complete chunk content by id from Pinecone.

        Args:
            id: Chunk id (UUID stored when indexing).

        Returns:
            {
                "id": id,
                "text": full_text,
                "metadata": metadata,
                "source_url": source_url,
            }

        Raises:
            ValueError: If id is missing or not found in Pinecone.
        """
        if not id:
            raise ValueError("Chunk id is required")

        logger.info(
            f"Fetching chunk from Pinecone: id={id}, "
            f"namespace='{PINECONE_NAMESPACE or '__default__'}'"
        )

        kwargs = {"ids": [id], "include_metadata": True}
        if PINECONE_NAMESPACE:
            kwargs["namespace"] = PINECONE_NAMESPACE

        try:
            resp = index.fetch(**kwargs)
        except NotFoundException:
            raise ValueError(f"Chunk with id '{id}' not found in Pinecone")

        vecs = resp.vectors or {}
        if id not in vecs:
            raise ValueError(f"Chunk with id '{id}' not found in Pinecone")

        vec = vecs[id]
        meta = vec.metadata or {}
        text = meta.get("text", "")
        source_url = meta.get("source_url")

        return {
            "id": id,
            "text": text,
            "source_url": source_url,
            "metadata": meta,
        }

    return mcp


def main():
    """Main function to start the MCP server."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found.")
        raise ValueError("OPENAI_API_KEY is required")

    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not found.")
        raise ValueError("PINECONE_API_KEY is required")

    if not PINECONE_INDEX_HOST:
        logger.error("PINECONE_INDEX_HOST not found.")
        raise ValueError(
            "PINECONE_INDEX_HOST is required (use the index host from Pinecone console)."
        )

    logger.info(
        f"Starting MCP server with Pinecone host='{PINECONE_INDEX_HOST}', "
        f"namespace='{PINECONE_NAMESPACE or '__default__'}', "
        f"embed_model='{EMBED_MODEL}', dim={INDEX_DIM}"
    )

    server = create_server()

    logger.info("Starting MCP server on port 8000 (HTTP transport)")
    try:
        server.run(transport="http", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
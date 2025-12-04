# MCP Knowledge Base Server

## Overview
This project is a Model Context Protocol (MCP) server that provides semantic search capabilities over a Pinecone vector database. It uses OpenAI embeddings to enable intelligent search through knowledge base content.

## Purpose
- **MCP Server**: Exposes semantic search and document retrieval tools via the Model Context Protocol
- **Knowledge Base Indexing**: Includes a utility to crawl and index documentation into Pinecone

## Components

### 1. mcp_server.py
The main MCP server application that exposes two tools:
- `search(query, top_k)`: Performs semantic search using OpenAI embeddings and Pinecone
- `fetch(id)`: Retrieves complete chunk content by ID from Pinecone

The server runs on port 8000 using HTTP transport.

### 2. reindex_productfruits_kb.py
A utility script that:
- Crawls the ProductFruits knowledge base website
- Extracts and chunks text content
- Generates embeddings using OpenAI
- Stores vectors in Pinecone for semantic search

## Configuration

### Required Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for generating embeddings
- `PINECONE_API_KEY`: Pinecone API key for vector database access
- `PINECONE_INDEX_HOST`: Pinecone index host (e.g., "your-index-xxxx.svc.pinecone.io")

### Optional Environment Variables
- `PINECONE_NAMESPACE`: Namespace within Pinecone index (defaults to "" for default namespace)
- `EMBED_MODEL`: OpenAI embedding model (defaults to "text-embedding-3-large")
- `INDEX_DIM`: Embedding dimensions (defaults to 3072)

## Project Architecture

### Technology Stack
- **Python 3.12**: Core runtime
- **FastMCP**: MCP server framework
- **OpenAI**: Embedding generation
- **Pinecone**: Vector database for semantic search
- **BeautifulSoup4**: HTML parsing for web crawling
- **Requests**: HTTP client for web crawling

### Data Flow
1. Documents are crawled and chunked
2. Text chunks are embedded using OpenAI
3. Embeddings are stored in Pinecone with metadata
4. MCP server provides search interface
5. Queries are embedded and matched against Pinecone vectors

## Recent Changes
- **2024-12-04**: Initial project setup in Replit environment
  - Installed all Python dependencies
  - Configured workflow to run MCP server
  - Set up environment variable management

## Usage

### Running the MCP Server
The server starts automatically via the configured workflow. It listens on port 8000 for MCP protocol connections.

### Reindexing the Knowledge Base
To reindex the ProductFruits knowledge base:
```bash
python reindex_productfruits_kb.py
```

Note: Ensure all required environment variables are set before running.

## Development Notes
- The server uses stateless HTTP transport for MCP
- Embedding model defaults to text-embedding-3-large (3072 dimensions)
- Chunking parameters: MAX_CHUNK_CHARS=2000, MIN_CHUNK_CHARS=200
- Crawling limit: MAX_PAGES=300

# Basic RAG System

RAG implementation with ChromaDB and OpenAI. Processes text documents, generates embeddings, and provides AI-powered question answering.

Based on the [RAG Crash Course tutorial](https://www.youtube.com/watch?v=ea2W8IogX80&t=1437s), enhanced with custom implementations and improved error handling.

## Features

- Document loading and chunking
- OpenAI embeddings (`text-embedding-3-small`)
- ChromaDB vector storage
- Context-aware responses

## Installation

```bash
cd basic_rag
uv install
```

Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
uv run main.py
```

Places text documents (`.txt`) in `news_articles/` directory.

## Custom Usage

```python
from embedding import DocumentEmbedder
from chroma import ChromaDb

embedder = DocumentEmbedder("./documents/")
chunks = embedder.get_chunks()

db = ChromaDb()
collection = db.create_collection("my_collection")
db.add_chunks(chunks, "my_collection")
results = db.query_documents("Your question", "my_collection")
```

## Configuration

- **Embedding model**: Modify `model_name` in `chroma.py`
- **Chunk size/overlap**: Adjust in `embedding.py` (`chunk_size=1000`, `chunk_overlap=20`)
- **LLM model**: Change in `main.py`

## Troubleshooting

- **Dimension mismatch**: Use same embedding model for creation and querying
- **API key issues**: Verify `.env` file and API credits
- **Collection not found**: Create collection before adding chunks

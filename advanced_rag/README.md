# Advanced RAG Techniques

Advanced RAG implementation comparing HyDE (Hypothetical Document Embeddings) and Multi-Query Expansion using ChromaDB and OpenAI.

## Techniques

- **HyDE**: Generates hypothetical answers to bridge vocabulary gaps between queries and documents
- **Multi-Query Expansion**: Generates multiple related subqueries for comprehensive retrieval with automatic deduplication

## Features

- PDF processing and chunking
- Persistent ChromaDB storage
- Smart deduplication
- CLI interface with progress tracking

## Installation

```bash
cd advanced_rag
uv install
```

Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

Place PDF in `data/` directory as `microsoft-annual-report.pdf`.

## Usage

```bash
python main.py
```

Modify the question in `main.py` to test custom queries.

## Configuration

- **Embeddings**: `text-embedding-3-small`
- **Text Generation**: `gpt-4.1-nano`
- **Results per query**: 5

**Note**: System prompts are optimized for financial reports. Modify prompts in `response.py` for other document types.

## Project Structure

- `main.py` - Demo script
- `chroma.py` - ChromaDB wrapper
- `pdf_processor.py` - PDF processing
- `response.py` - Response generation
- `util.py` - Utilities

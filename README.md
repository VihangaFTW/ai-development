# AI Development Portfolio

Collection of AI projects showcasing RAG techniques, LangGraph agents, and AI application development.

## Projects

### Advanced RAG (`advanced_rag/`)
Professional RAG with HyDE and Multi-Query Expansion. Features ChromaDB integration, PDF processing, and smart deduplication.

### Basic RAG (`basic_rag/`)
Foundational RAG system for learning. Includes document processing, vector storage, and semantic search.

### LangGraph Agents (`langgraph_agents/`)
AI agents with tool usage and memory. Includes conversational agents, RAG-powered assistants, and specialized bots.

### LangGraph Basics (`langgraph_basics/`)
Interactive tutorials covering sequential graphs, conditional logic, looping, and multiple inputs.

## Quick Start

```bash
git clone https://github.com/VihangaFTW/ai-development
cd advanced_rag/  # or basic_rag/, langgraph_agents/, langgraph_basics/
uv sync
```

## Development Setup

```bash
uv sync
uv run pre-commit install
```

Pre-commit hooks automatically format code with `ruff` on every commit.

## Technologies

- **LangGraph/LangChain** - Agent orchestration
- **OpenAI API** - Models and embeddings
- **ChromaDB** - Vector database
- **Python 3.13+** - Primary language
- **UV** - Package manager

## Learning Path

1. `langgraph_basics/` - Fundamentals
2. `basic_rag/` - RAG concepts
3. `langgraph_agents/` - Agent systems
4. `advanced_rag/` - Optimized RAG implementation



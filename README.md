# 🤖 AI Development Portfolio

A (to-be) comprehensive collection of AI projects showcasing modern techniques in **Retrieval-Augmented Generation (RAG)**, **LangGraph Agents**, and **AI application development**. Monthly contributions to this repo are expected unless I'm busy with work :)

## 🚀 Projects Overview

### 📚 Advanced RAG Techniques (`advanced_rag/`)

**Professional RAG implementation with advanced query expansion techniques**

- **🔍 HyDE (Hypothetical Document Embeddings)**: Query expansion using AI-generated hypothetical answers
- **🎯 Multi-Query Expansion**: Batch processing with multiple related subqueries
- **💾 ChromaDB Integration**: Persistent vector storage with OpenAI embeddings
- **🤖 Smart Deduplication**: Automatic removal of duplicate results
- **🎨 Professional CLI**: Beautiful demo interface with progress tracking
- **📄 PDF Processing**: Automated document chunking and embedding generation

**Tech Stack**: ChromaDB, OpenAI API, Python

> **Note**
>
> A de-noiser pipeline for the llm generated subqueries has not been implemented yet. This section will be updated once the implementation is complete.

### 📖 Basic RAG Implementation (`basic_rag/`)

**Foundational RAG system for learning and experimentation**

- **🔧 Core RAG Pipeline**: Essential retrieval-augmented generation workflow
- **📰 News Article Processing**: Text processing and embedding for news content
- **🗄️ Vector Storage**: Basic ChromaDB implementation
- **🔍 Semantic Search**: Document similarity and retrieval
- **📊 Embedding Utilities**: Text vectorization and similarity matching

**Tech Stack**: ChromaDB, Python, OpenAI API and their embedding models

### 🤖 LangGraph Agents (`langgraph_agents/`)

**Intelligent AI agents with tool usage and memory capabilities**

- **🧠 Agent Bot**: General-purpose conversational agent
- **🐾 Agent Cuddles**: Specialized interactive agent
- **📊 Agent RAG**: RAG-powered intelligent assistant
- **💭 Memory-Enabled Chatbot**: Persistent conversation memory
- **🛠️ Tool-Using Chatbot**: Multi-tool integration and function calling
- **📈 Stock Market Analysis**: Financial document processing agent

**Tech Stack**: LangGraph, LangChain, OpenAI API

### 🎓 LangGraph Basics (`langgraph_basics/`)

**Interactive tutorials and foundational concepts**

- **👋 Hello World**: Basic LangGraph introduction
- **🔄 Sequential Graphs**: Linear workflow implementation
- **🔀 Conditional Logic**: Branching and decision-making workflows
- **🔁 Looping Graphs**: Iterative processing patterns
- **📥 Multiple Inputs**: Complex input handling
- **📓 Jupyter Notebooks**: Interactive learning materials

**Tech Stack**: LangGraph, Jupyter, Python

## 🛠️ Technologies Used

### **Core AI/ML**

- **LangGraph** - Agent workflow orchestration
- **LangChain** - AI application framework
- **OpenAI API** - Language models and embeddings
- **ChromaDB** - Vector database for embeddings

### **Development Tools**

- **Python 3.13+** - Primary programming language
- **UV** - Fast Python package manager
- **Jupyter Notebooks** - Interactive development
- **PDF Processing** - Document parsing and chunking

### **Key Techniques**

- **RAG (Retrieval-Augmented Generation)** - Context-aware AI responses
- **Vector Embeddings** - Semantic search and similarity
- **Agent Architectures** - Autonomous AI workflows
- **Tool Integration** - Function calling and external APIs
- **Memory Management** - Persistent conversation state

## 🚀 Quick Start

Each project includes its own detailed README with setup instructions:

```bash
# Clone the repository
git clone https://github.com/VihangaFTW/ai-development

# Choose your project of interest
cd advanced_rag/     # For advanced RAG techniques
cd basic_rag/        # For basic RAG learning
cd langgraph_agents/ # For intelligent agents
cd langgraph_basics/ # For LangGraph tutorials
```

## 🛠️ Development Setup

**For developers working on this repository:**

After cloning the repository, set up the development environment:

```bash
# Install dependencies using uv (our preferred package manager)
uv sync

# Install pre-commit hooks for automatic code formatting
uv run pre-commit install
```

**What this does:**

- 🔧 **Code Formatting**: Automatically formats Python code with `ruff` on every commit
- 🧹 **Code Linting**: Fixes common code issues automatically
- ✅ **Quality Assurance**: Ensures consistent code style across the project

**Note**: Each contributor must run `uv run pre-commit install` once after cloning to enable automatic code formatting.

## 📊 Project Complexity

| Project             | Complexity      | Focus Area       | Best For                        |
| ------------------- | --------------- | ---------------- | ------------------------------- |
| `langgraph_basics/` | 🟢 Beginner     | Learning         | Getting started with LangGraph  |
| `basic_rag/`        | 🟡 Intermediate | RAG Fundamentals | Understanding RAG concepts      |
| `advanced_rag/`     | 🟡 Intermediate | RAG Techniques   | Optimizing RAG implementation   |
| `langgraph_agents/` | 🟡 Intermediate | Agent Systems    | Building intelligent agents     |

## 💡 Learning Path

**Recommended progression for newcomers:**

1. **Start with `langgraph_basics/`** - Learn fundamental concepts
2. **Explore `basic_rag/`** - Understand retrieval-augmented generation
3. **Try `langgraph_agents/`** - Build intelligent agent systems
4. **Master `advanced_rag/`** - Implement production-ready RAG systems

## 🎯 Use Cases

- **Document Q&A Systems** - Query large document collections
- **Intelligent Chatbots** - Context-aware conversational AI
- **Research Assistants** - AI-powered information retrieval
- **Content Analysis** - Automated document processing
- **Financial Analysis** - AI agents for market research
- **Educational Tools** - Interactive learning systems

## 📄 License

This project is licensed under the MIT License - see individual project folders for specific details.

---

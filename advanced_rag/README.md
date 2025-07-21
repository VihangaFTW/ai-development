# 🚀 Advanced RAG Techniques Demo

A comprehensive demonstration of advanced Retrieval-Augmented Generation (RAG) techniques using ChromaDB and OpenAI. This project implements and compares two sophisticated query expansion methods: **HyDE (Hypothetical Document Embeddings)** and **Multi-Query Expansion**.

## 🎯 Overview

This project showcases advanced RAG techniques that go beyond basic semantic search to provide more accurate and comprehensive document retrieval. By expanding user queries through different methodologies, we achieve better semantic matching and more relevant context for AI-generated responses.

### 🧩 Implemented Techniques

#### 1. **HyDE (Hypothetical Document Embeddings)**

- Generates a hypothetical answer to the user's question using an LLM
- Combines the original question with the hypothetical answer for enhanced retrieval
- Bridges vocabulary gaps between questions and document terminology
- Particularly effective for domain-specific queries where question and answer vocabularies differ

#### 2. **Multi-Query Expansion**

- Generates multiple related subqueries from the original question
- Performs batch retrieval using all queries simultaneously
- Captures different aspects and perspectives of the user's question
- Includes automatic deduplication to avoid redundant results
- Leverages ChromaDB's efficient batch processing capabilities

## ✨ Features

- **🎨 CLI Interface**: Professional demo presentation with progress indicators and emojis
- **📄 PDF Processing**: Automatic document chunking and embedding generation
- **🗄️ Vector Database**: Persistent ChromaDB storage with OpenAI embeddings
- **🔍 Duplicate Handling**: Smart deduplication for multi-query results
- **🤖 Contextual Responses**: AI-generated answers grounded in retrieved documents
- **📊 Technique Comparison**: Side-by-side demonstration of both RAG approaches
- **🛡️ Error Handling**: Graceful handling of edge cases and missing data

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- UV package manager (recommended)

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/VihangaFTW/ai-development
   cd advanced_rag
   ```

2. **Install dependencies using UV**:

   ```bash
   uv install
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Add your document**:
   Place your PDF document in the `data/` directory as `microsoft-annual-report.pdf`

## 🚀 Usage

### Basic Demo

Run the complete demo showcasing both RAG techniques:

```bash
python main.py
```

This will execute a comprehensive demonstration showing:

1. HyDE technique with step-by-step progress
2. Multi-Query Expansion technique with batch processing
3. Side-by-side comparison of results

### Individual Techniques

You can also run techniques individually by modifying the `main()` function:

```python
# Run only HyDE technique
run_expanded_single_query(db, "Your question here")

# Run only Multi-Query technique
run_expanded_multiple_queries(db, "Your question here")
```

### Custom Questions

Modify the question in `main.py`:

```python
question = "What were Microsoft's key financial metrics for 2023?"
```

## 📁 Project Structure

```
advanced_rag/
├── main.py                 # Main demo script with CLI interface
├── chroma.py              # ChromaDB wrapper with singleton pattern
├── pdf_processor.py       # PDF document processing and chunking
├── response.py            # AI response generation functions
├── util.py                # Utility functions (word wrap, API key loading)
├── data/                  # Document storage directory
│   └── microsoft-annual-report.pdf
├── pyproject.toml         # UV package configuration
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## 🔧 Technical Architecture

### Core Components

1. **ChromaDb Class** (`chroma.py`)

   - Singleton pattern implementation
   - OpenAI embedding integration
   - Persistent vector storage
   - Query deduplication support

2. **PDFChunkGenerator** (`pdf_processor.py`)

   - Document parsing and chunking
   - Metadata extraction
   - Chunk ID generation

3. **Response Generation** (`response.py`)

   - HyDE hypothetical answer generation
   - Multi-query expansion
   - Context-aware response synthesis

4. **CLI Interface** (`main.py`)
   - Professional demo presentation
   - Progress tracking
   - Error handling and user feedback

### RAG Pipeline Flow

#### HyDE Technique (6 Steps)

1. 📂 Set up ChromaDB collection
2. 📄 Process PDF document
3. 💾 Store chunks in vector database
4. 🤖 Generate hypothetical answer
5. 🔗 Combine query with hypothetical answer
6. 🔎 Search vector database

#### Multi-Query Technique (7 Steps)

1. 📂 Set up ChromaDB collection
2. 📄 Process PDF document
3. 💾 Store chunks in vector database
4. 🤖 Generate multiple related subqueries
5. 🔗 Combine all queries for batch processing
6. 🔎 Perform batch search with deduplication
7. 🧠 Generate comprehensive response

## 🎛️ Configuration

### Model Settings

Default models used:

- **Embeddings**: `text-embedding-3-small`
- **Text Generation**: `gpt-4.1-nano`

### System Prompts

**⚠️ Important Note**: The current system prompts are specifically optimized for **financial reports and annual reports**. The prompts include:

- Financial research assistant persona
- Instructions to look for financial metrics, revenue, profit data
- Context understanding for corporate financial documents
- Examples tailored to annual report structure

#### Customizing for Other Document Types

To adapt this project for non-financial documents, you'll need to modify the system prompts in `response.py`:

1. **HyDE Generation** (`generate_single_query_response`):

   ```python
   system_prompt = """You are a helpful expert [DOMAIN] research assistant.
   Provide an example answer to the given question, that might be found in a document like [DOCUMENT_TYPE]."""
   ```

2. **Multi-Query Expansion** (`generate_multi_query_response`):

   ```python
   system_prompt = """You are a knowledgeable [DOMAIN] research assistant.
   Your users are inquiring about [DOCUMENT_TYPE].
   For the given question, propose up to five related questions..."""
   ```

3. **Response Generation** (`generate_response_with_context`):
   ```python
   system_prompt = """You are a helpful [DOMAIN] research assistant.
   Answer the user's question based solely on the provided context from [DOCUMENT_TYPE]..."""
   ```

### Retrieval Parameters

- **Number of results**: 5 per query
- **Deduplication**: Enabled by default
- **Word wrap width**: 120 characters

### ChromaDB Settings

- **Storage**: Persistent local storage (`./chroma`)
- **Collections**: Separate collections for each technique
- **Embedding function**: OpenAI with API key

## 🔍 Example Output

```
🌟══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════🌟
                    🎯 ADVANCED RAG TECHNIQUES COMPARISON DEMO 🎯
                         Powered by ChromaDB + OpenAI
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📖 Document: Microsoft Annual Report 2023
🧩 Techniques: HyDE vs Multi-Query Expansion
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🔍══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════🔍
🚀 ADVANCED RAG DEMO - HyDE (Hypothetical Document Embeddings) Technique
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📝 QUESTION: What details can you provide about the factors that led to revenue growth?
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📂 Step 1/6: Setting up ChromaDB collection...
✅ Collection created/retrieved successfully
...
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

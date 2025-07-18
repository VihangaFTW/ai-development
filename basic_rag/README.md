# RAG System with ChromaDB and OpenAI

A Retrieval-Augmented Generation (RAG) system that combines document retrieval with AI-powered question answering. This system processes text documents, creates semantic embeddings, stores them in a vector database, and provides intelligent responses to user queries.

## 📚 About This Project

This RAG implementation follows the core concepts taught in the [RAG Crash Course YouTube tutorial](https://www.youtube.com/watch?v=ea2W8IogX80&t=1437s), but has been significantly modified and enhanced for experimentation and additional features. The codebase includes custom implementations, improved error handling, comprehensive documentation, and a cleaner CLI interface while maintaining the fundamental RAG architecture principles from the course.

## 🚀 Features

- **Document Processing**: Automatically loads and chunks text documents
- **Vector Embeddings**: Uses OpenAI's `text-embedding-3-small` model for high-quality embeddings
- **Vector Database**: ChromaDB with persistent storage for fast semantic search
- **Intelligent Responses**: OpenAI model powered answers using retrieved context
- **Clean CLI Interface**: Progress tracking and user-friendly output

## 📋 Requirements

- Python 3.13+
- OpenAI API key
- uv package manager

## 🛠️ Installation

1. **Clone the repository and navigate to the RAG directory**

   ```bash
   cd rag/
   ```

2. **Install dependencies using uv**

   ```bash
   uv install
   ```

3. **Set up your OpenAI API key**

   Create a `.env` file in the rag directory:

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## 📁 Project Structure

```
rag/
├── main.py              # Main RAG pipeline orchestration
├── embedding.py         # Document loading, chunking, and embedding generation
├── chroma.py           # ChromaDB vector database operations
├── util.py             # Utility functions (API key loading)
├── news_articles/      # Directory containing text documents
├── chroma/             # ChromaDB persistent storage (auto-created)
├── pyproject.toml      # Project dependencies
├── uv.lock            # Dependency lock file
└── README.md          # This file
```

## 🚀 Usage

### Basic Usage

Run the complete RAG pipeline:

```bash
uv run main.py
```

This will:

1. Load documents from the `news_articles/` directory
2. Create document chunks with overlapping content
3. Generate embeddings for each chunk
4. Store embeddings in ChromaDB
5. Query the database with a sample question
6. Generate an AI-powered response

### Custom Implementation

You can also use the components individually:

```python
from embedding import DocumentEmbedder
from chroma import ChromaDb

# Process documents
embedder = DocumentEmbedder("./your_documents/")
chunks = embedder.get_chunks()

# Set up vector database
vector_db = ChromaDb()
collection = vector_db.create_collection("my_collection")
vector_db.add_chunks(chunks, "my_collection")

# Query the database
relevant_chunks = vector_db.query_documents("Your question here", "my_collection")
```

## 📖 API Reference

### DocumentEmbedder Class

Handles document loading, chunking, and embedding generation.

```python
embedder = DocumentEmbedder(path="./documents/")
chunks = embedder.get_chunks()
```

### ChromaDb Class

Manages vector database operations with ChromaDB.

```python
db = ChromaDb(storage_path="./chroma")
collection = db.create_collection("collection_name")
db.add_chunks(chunks, "collection_name")
results = db.query_documents("question", "collection_name", n_results=5)
```

## ⚙️ Configuration

### Embedding Model

The system uses OpenAI's `text-embedding-3-small` by default. To change models, modify the `model_name` parameter in `chroma.py`:

```python
self.ef = OpenAIEmbeddingFunction(
    api_key=load_and_get_key(),
    model_name='text-embedding-3-large'  # Change here
)
```

### Chunking Parameters

Adjust chunk size and overlap in `embedding.py`:

```python
def __chunk_generator(self, text: str, chunk_size: int=1000, chunk_overlap: int=20):
```

### LLM Model

Change the response generation model in `main.py`:

```python
llm_response = client.chat.completions.create(
    model='gpt-4',  # Change here
    messages=[...]
)
```

## 📄 Document Format

Place your text documents (`.txt` files) in the `news_articles/` directory or specify a custom path. The system will automatically:

- Load all `.txt` files from the directory
- Split them into manageable chunks
- Generate embeddings for semantic search

## 🔧 Troubleshooting

### Common Issues

1. **Dimension Mismatch Error**

   - Ensure the same embedding model is used for both creation and querying
   - Delete the `chroma/` directory and recreate collections if needed

2. **API Key Issues**

   - Verify your `.env` file contains the correct OpenAI API key
   - Check that the key has sufficient credits

3. **Collection Not Found**
   - Make sure to create the collection before adding chunks
   - Verify collection names match exactly (case-sensitive)

## 📊 Example Output

```
Starting RAG system...
Loaded 18 documents
Processing 05-03-ai-powered-supply-chain-startup-pando-lands-30m-investment.txt...
  Generated 15 chunks
Generating embeddings for 67 chunks...
  Progress: 5/67 embeddings generated
  Progress: 10/67 embeddings generated
  ...
Setting up vector database...
Database setup complete

Querying: Has Slack started prioritizing ai features in the app?

Yes, Slack has started prioritizing AI features, integrating AI into its platform
with products like SlackGPT and partnerships with OpenAI. They are working to embed
AI to enhance user experience, communication, and workflow automation.
However, many of these features are still in development or in beta.
```

## 📝 License

This project is for educational and development purposes.

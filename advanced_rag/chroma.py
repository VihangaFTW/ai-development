from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddable
from chromadb.api.models.Collection import Collection, QueryResult
from typing import cast
from util import load_and_get_key


class ChromaDb():
    """
    A singleton wrapper class for ChromaDB vector database operations with OpenAI embeddings.
    
    This class implements the singleton pattern to ensure only one database connection exists
    throughout the application lifecycle. It provides a simplified interface for creating 
    collections, storing document chunks, and querying the ChromaDB vector database. 
    Uses OpenAI's text-embedding-3-small model for generating embeddings and supports 
    persistent storage of vector data.
    
    Singleton Behavior:
        - Only one instance of ChromaDb can exist per application
        - All calls to ChromaDb() return the same instance
        - Database connection and embedding function are shared across the application
        - Thread-safe initialization ensures proper setup in multi-threaded environments
    
    Class Attributes:
        _instance (ChromaDb | None): Stores the singleton instance of the class.
        _initialized (bool): Flag to track whether the singleton has been initialized.
    
    Instance Attributes:
        client (chromadb.PersistentClient): ChromaDB persistent client for database operations.
        ef (OpenAIEmbeddingFunction): OpenAI embedding function for generating text embeddings.
    
    Example:
        >>> db1 = ChromaDb()
        >>> db2 = ChromaDb()  # Returns the same instance as db1
        >>> print(db1 is db2)  # True
        >>> collection = db1.create_collection("my-collection")
        >>> db2.add_chunks(chunk_ids, chunks, "my-collection")  # Uses same connection
    """
    # singleton pattern implementation
    _instance =  None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # first time: create a new instance
            cls._instance = super().__new__(cls)
            
        return cls._instance
    
    
    
    def __init__(self, storage_path: str = "./chroma") -> None:
        """
        Initialize the ChromaDB client with persistent storage and OpenAI embeddings.
        
        Args:
            storage_path (str, optional): Path to the directory where ChromaDB will store
                                        persistent data. Defaults to "./chroma".
        """
        # skip if an instance has been already initialized
        if self._initialized:
            return        
        
        # initialize the chroma client with persistent storage
        self.client = chromadb.PersistentClient(storage_path)
        # use openai's embedding llm instead of the default embedding llm provided by chromadb
        self.ef = OpenAIEmbeddingFunction(api_key = load_and_get_key(), model_name='text-embedding-3-small')
        self._initialized =  True
        
        
    def create_collection(self, collection_name: str, metadata: dict[str, str] | None = None) -> Collection:
        """
        Create or retrieve a ChromaDB collection with OpenAI embeddings.
        
        This method creates a new collection or retrieves an existing one with the same name.
        The collection is configured to use OpenAI's text-embedding-3-small model for
        generating embeddings.
        
        Args:
            collection_name (str): The name of the collection to create or retrieve.
            metadata (dict[str, str] | None, optional): Optional metadata to associate
                                                       with the collection. Defaults to None.
        
        Returns:
            Collection: The ChromaDB collection object for storing and querying documents.
        """
        return self.client.get_or_create_collection(
                        name=collection_name,
                        #? the cast is to fix a type checker bug. Alternative is to comment the line with "# type: ignore"
                        embedding_function= cast(EmbeddingFunction[Embeddable],self.ef),
                        metadata=metadata
                    )
        

    def add_chunks(self, chunk_ids: list[str], chunks: list[str], collection_name: str, **kwargs) -> None:
        """
        Add document chunks to a ChromaDB collection.
        
        This method takes lists of chunk IDs and document content and stores them in the specified
        collection. The embeddings are automatically generated using the configured OpenAI embedding function.
        
        Args:
            chunk_ids (list[str]): List of unique identifiers for each document chunk.
            chunks (list[str]): List of document content strings to store.
            collection_name (str): The name of the collection where chunks will be stored.
            **kwargs: Additional parameters passed to ChromaDB's upsert method:
                - embeddings (list[list[float]], optional): Pre-computed embeddings if not using embedding function
                - metadatas (list[dict], optional): Metadata dictionaries for each document
        
        Returns:
            None: Stores the chunks in the database or prints an error if collection not found.
        
        Raises:
            ValueError: If the specified collection does not exist in the database.
        
        Example:
            >>> db = ChromaDb()
            >>> db.add_chunks(
            ...     chunk_ids=["doc1_chunk1", "doc1_chunk2"],
            ...     chunks=["First chunk content", "Second chunk content"],
            ...     collection_name="my_collection",
            ...     metadatas=[{"source": "doc1"}, {"source": "doc1"}]
            ... )
        """
        
        # retrieve the collection
        try:
            collection: Collection = self.client.get_collection(name=collection_name)
            collection.upsert(ids=chunk_ids, documents=chunks, **kwargs)
                
        except ValueError as e:
            print(f'Error: Collection "{collection_name}" not found')
            
    def query_documents(self, question: str, collection_name: str, n_results=2, **kwargs) -> QueryResult | list[str] | None:
        """
        Query the ChromaDB collection for documents most relevant to a given question.
        
        This method uses semantic similarity search to find document chunks that are most
        relevant to the input question. It generates an embedding for the question using
        the same OpenAI model used for document embeddings and returns the results based 
        on the 'include' parameter.
        
        Args:
            question (str): The question or query text to search for relevant documents.
            collection_name (str): The name of the collection to search within.
            n_results (int, optional): The maximum number of relevant chunks to return.
                                     Defaults to 2.
            **kwargs: Additional parameters passed to ChromaDB's query method:
                - where (dict, optional): Metadata filtering conditions
                - where_document (dict, optional): Document content filtering conditions  
                - include (list[str], optional): What to include in results 
                  (e.g., ["documents", "distances", "embeddings"])
                  If not specified, defaults to ["documents"] and returns list[str]
        
        Returns:
            QueryResult | list[str] | None: 
                - If 'include' is specified in kwargs: Returns the full QueryResult object
                - If 'include' not specified (default): Returns list[str] of document chunks
                - Returns None if collection is not found
        
        Raises:
            ValueError: If the specified collection does not exist in the database.
        
        Examples:
            >>> db = ChromaDb()
            
            # Default behavior - returns list of document strings
            >>> docs = db.query_documents("What is revenue?", "docs")
            >>> print(docs)  # ["Revenue was $50B", "Total revenue increased..."]
            
            # Custom include - returns full QueryResult
            >>> results = db.query_documents(
            ...     "What is revenue?", "docs",
            ...     include=["documents", "distances"]
            ... )
            >>> print(results['documents'])  # Document content
            >>> print(results['distances'])  # Similarity scores
        """
        try:
            collection: Collection = self.client.get_collection(name=collection_name)
            
            # check if user specified custom 'include' parameter
            include_param = kwargs.get('include')
            
            # if no custom include specified, default to documents only for backward compatibility
            if not include_param:
                kwargs['include'] = ['documents']
                results: QueryResult = collection.query(query_texts=question, n_results=n_results, **kwargs)
                
                # extract and flatten documents for backward compatibility
                documents = results.get('documents', [])
                if documents:
                    relevant_chunks: list[str] = [doc for sublist in documents for doc in sublist]
                else:
                    relevant_chunks = []
                
                return relevant_chunks
            else:
                # uer specified custom include, return full QueryResult
                results: QueryResult = collection.query(query_texts=question, n_results=n_results, **kwargs)
                return results
        
        except ValueError as e:
            print(f'Error: Collection "{collection_name}" not found')
            return None
            
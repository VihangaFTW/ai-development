from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddable
from chromadb.api.models.Collection import Collection, QueryResult
from typing import cast
from util import load_and_get_key
from embedding import Chunk


class ChromaDb():
    """
    A wrapper class for ChromaDB vector database operations with OpenAI embeddings.
    
    This class provides a simplified interface for creating collections, storing document chunks,
    and querying the ChromaDB vector database. It uses OpenAI's text-embedding-3-small model
    for generating embeddings and supports persistent storage of vector data.
    
    Attributes:
        client: ChromaDB persistent client for database operations.
        ef: OpenAI embedding function for generating text embeddings.
    """
    
    def __init__(self, storage_path: str = "./chroma") -> None:
        """
        Initialize the ChromaDB client with persistent storage and OpenAI embeddings.
        
        Args:
            storage_path (str, optional): Path to the directory where ChromaDB will store
                                        persistent data. Defaults to "./chroma".
        """
        # initialize the chroma client with persistent storage
        self.client = chromadb.PersistentClient(storage_path)
        # use openai's embedding llm instead of the default embedding llm provided by chromadb
        self.ef = OpenAIEmbeddingFunction(api_key = load_and_get_key(), model_name='text-embedding-3-small')
        
        
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
        

    def add_chunks(self, chunks: list[Chunk], collection_name: str) -> None:
        """
        Add document chunks with their embeddings to a ChromaDB collection.
        
        This method takes a list of processed document chunks and stores them in the specified
        collection. Each chunk includes its ID, content, and pre-generated embedding vector.
        
        Args:
            chunks (list[Chunk]): List of document chunks to store. Each chunk should contain
                                'chunk_id', 'chunk_content', and optionally 'chunk_embedding'.
            collection_name (str): The name of the collection where chunks will be stored.
        
        Returns:
            None: Stores the chunks in the database or prints an error if collection not found.
        
        Raises:
            ValueError: If the specified collection does not exist in the database.
        """
        
        # retrieve the collection
        try:
            collection: Collection = self.client.get_collection(name=collection_name)
            for chunk in chunks:
                collection.upsert(ids=chunk['chunk_id'], documents=chunk['chunk_content'], embeddings=chunk.get("chunk_embedding", None))
                
        except ValueError as e:
            print(f'Error: Collection "{collection_name}" not found')
            
    def query_documents(self, question: str, collection_name: str, n_results=2) -> list[str] | None:
        """
        Query the ChromaDB collection for documents most relevant to a given question.
        
        This method uses semantic similarity search to find document chunks that are most
        relevant to the input question. It generates an embedding for the question using
        the same OpenAI model used for document embeddings and returns the most similar chunks.
        
        Args:
            question (str): The question or query text to search for relevant documents.
            collection_name (str): The name of the collection to search within.
            n_results (int, optional): The maximum number of relevant chunks to return.
                                     Defaults to 2.
        
        Returns:
            list[str] | None: A list of relevant document chunks as strings, or None if
                            the collection is not found. Returns an empty list if no
                            relevant documents are found.
        
        Raises:
            ValueError: If the specified collection does not exist in the database.
        """
        try:
            collection: Collection = self.client.get_collection(
                name=collection_name, 
                embedding_function=cast(EmbeddingFunction[Embeddable], self.ef)
            )
            results: QueryResult = collection.query(query_texts=question, n_results=n_results)
            
            documents = results.get('documents', [])
            if documents:
                relevant_chunks: list[str] = [doc for sublist in documents for doc in sublist]
            else:
                relevant_chunks = []
            
            return relevant_chunks
        
        except ValueError as e:
            print(f'Error: Collection "{collection_name}" not found')
            return None
            
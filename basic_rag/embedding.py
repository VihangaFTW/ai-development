from typing import TypedDict, NotRequired
from util import load_and_get_key
from openai import OpenAI
import os


class Chunk(TypedDict):
    chunk_id: str
    chunk_content: str
    chunk_embedding: NotRequired[list[float]]


class Document(TypedDict):
    doc_name: str
    doc_content: str


openai_key = load_and_get_key()

openai_client = OpenAI(api_key=openai_key)


class DocumentEmbedder:
    """
    A class for loading documents, splitting them into chunks, and generating embeddings.

    This class handles the complete pipeline from loading text documents from a directory,
    splitting them into manageable chunks, and generating OpenAI embeddings for each chunk.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the DocumentEmbedder with a path.

        Args:
            path (str): The path to the directory containing documents to process.
        """
        self.path = path
        self.chunks: list[Chunk] = []

    def get_chunks(self) -> list[Chunk]:
        """
        Main method to get processed chunks with embeddings.

        This method orchestrates the entire pipeline: loading documents,
        chunking them, and generating embeddings for each chunk.

        Returns:
            list[Chunk]: A list of chunks with their content and embeddings.
        """

        # load documents
        documents = self.__load_documents(self.path)
        print(f"Loaded {len(documents)} documents")

        # generate chunks from all loaded documents
        self.chunks = self.__documents_to_chunks(documents)

        # associate the corresponding embedding for each of the generated chunks
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        for i, chunk in enumerate(self.chunks, 1):
            chunk["chunk_embedding"] = self.__get_openai_embedding(
                chunk["chunk_content"]
            )
            if i % 5 == 0 or i == len(self.chunks):
                print(f"  Progress: {i}/{len(self.chunks)} embeddings generated")

        return self.chunks

    def __load_documents(self, directory: str) -> list[Document]:
        """
        Load all text documents from the specified directory.

        Args:
            directory (str): Path to the directory containing text files.

        Returns:
            list[Document]: A list of documents with their names and content.
        """
        documents: list[Document] = []

        # loop through the files in given directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                # read the file contents
                with open(
                    os.path.join(directory, filename), "r", encoding="utf-8"
                ) as f:
                    documents.append({"doc_name": filename, "doc_content": f.read()})

        return documents

    def __chunk_generator(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 20
    ) -> list[str]:
        """
        Split text into overlapping chunks of specified size.

        Args:
            text (str): The text to be split into chunks.
            chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): Number of characters to overlap between chunks. Defaults to 20.

        Returns:
            list[str]: A list of text chunks.
        """
        chunks: list[str] = []
        start: int = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap

        return chunks

    def __documents_to_chunks(self, documents: list[Document]) -> list[Chunk]:
        """
        Convert a list of documents into chunks with unique identifiers.

        Args:
            documents (list[Document]): List of documents to be chunked.

        Returns:
            list[Chunk]: List of chunks with unique IDs and content.
        """
        document_chunks: list[Chunk] = []

        for doc in documents:
            print(f"Processing {doc['doc_name']}...")
            chunks: list[str] = self.__chunk_generator(doc["doc_content"])
            for i, chunk in enumerate(chunks):
                document_chunks.append(
                    {
                        "chunk_id": f"{doc['doc_name']}_chunk{i + 1}",
                        "chunk_content": chunk,
                    }
                )
            print(f"  Generated {len(chunks)} chunks")

        return document_chunks

    def __get_openai_embedding(self, text: str) -> list[float]:
        """
        Generate OpenAI embedding for the given text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        response = openai_client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        embedding: list[float] = response.data[0].embedding
        return embedding

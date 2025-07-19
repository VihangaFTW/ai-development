from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


class PDFChunkGenerator:
    """
    A generator class for processing PDF files and creating text chunks for RAG applications.
    
    This class extracts text content from PDF files and splits it into smaller chunks
    suitable for embedding and retrieval. The PDF is processed once during initialization
    and results are stored as instance attributes for efficient access.
    
    Attributes:
        pdf_path (str): The file path to the PDF document to be processed.
        texts (list[str] | None): Extracted text content from PDF pages.
        chunks (list[str]): Text chunks optimized for embedding and retrieval.
        chunk_ids (list[str]): Sequential IDs corresponding to each chunk.
        
    Raises:
        ValueError: If the provided PDF path does not exist.
        
    Example:
        >>> generator = PDFChunkGenerator("data/document.pdf")
        >>> print(f"Generated {len(generator.chunks)} text chunks")
        >>> for chunk_id, chunk in zip(generator.chunk_ids, generator.chunks):
        ...     print(f"Chunk {chunk_id}: {chunk[:50]}...")
    """
    
    def __init__(self, pdf_path: str) -> None:
        """
        Initialize the PDF chunk generator and process the document.
        
        Args:
            pdf_path (str): Path to the PDF file to process.
            
        Raises:
            ValueError: If the PDF path does not exist.
        """
        self.pdf_path = pdf_path
        
        if not os.path.exists(self.pdf_path):
            raise ValueError("PDF path does not exist")
        
        self.texts: list[str] | None = self.__pdf_to_texts()
        self.chunk_ids, self.chunks = self.__texts_to_chunks()
    
    
    def get_chunks(self) -> list[str]:
        """
        Get the processed text chunks suitable for embedding and retrieval.
        
        Returns the pre-computed chunks that were generated during initialization.
        These chunks are optimized for RAG applications with approximately 256 tokens each.
        
        Returns:
            list[str]: A list of text chunks ready for embedding and vector storage.
                      Each chunk is split at natural boundaries for optimal retrieval.
        """
        return self.chunks
    
    
    def get_chunk_ids(self) -> list[str]:
        """
        Get the sequential IDs for each text chunk.
        
        Returns the pre-computed chunk IDs that correspond to the chunks.
        Each ID represents the position of its corresponding chunk in the sequence as a string.
        
        Returns:
            list[str]: A list of sequential string IDs starting from "0", one for each chunk.
        """
        return self.chunk_ids
    
    
    def get_document_texts(self) -> list[str] | None:
        """
        Get the raw extracted text content from the PDF pages.
        
        Returns the pre-extracted text from each PDF page with empty pages removed.
        This is the unprocessed text before chunking.
        
        Returns:
            list[str] | None: A list of text strings, one for each non-empty page
                             in the PDF document, or None if extraction failed.
        """
        return self.texts
    
    
    def __pdf_to_texts(self) -> list[str] | None:
        """
        Extract and return all text content from the PDF pages.
        
        Reads the PDF file and extracts text from each page, removing any
        empty or whitespace-only pages. This method is called once during
        initialization.
        
        Returns:
            list[str] | None: A list of text strings, one for each non-empty page
                             in the PDF document, or None if an error occurs.
                      
        Raises:
            Prints error message if PDF reading fails, but doesn't raise exceptions
            to allow graceful degradation.
        """
        # read the pdf contents
        try:
            
            reader = PdfReader(self.pdf_path, strict=True)
            texts = [page.extract_text().strip() for page in reader.pages]
            # remove whitespace lines
            return [text for text in texts if text]
        except Exception as e:
            print(f'Error while parsing pdf: {e}')

    
    def __texts_to_chunks(self) -> tuple[list[str],list[str]]:
        """
        Generate text chunks with IDs suitable for embedding and retrieval.
        
        Takes the extracted PDF text and splits it into smaller chunks using
        RecursiveCharacterTextSplitter with tiktoken encoding. The chunks are
        optimized for RAG applications with a size of 256 tokens and no overlap.
        This method is called once during initialization.
        
        Returns:
            tuple[list[str], list[str]]: A tuple containing:
                - list[str]: Sequential chunk IDs as strings starting from "0"
                - list[str]: Text chunks, each approximately 256 tokens in size,
                           split at natural boundaries (paragraphs, sentences, etc.).
                           Returns empty lists if no texts were extracted.
                      
        """
        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=256, chunk_overlap=0)
        
        chunks: list[str] = token_splitter.split_text("\n\n".join(self.texts)) if self.texts else []
        
        chunk_ids =  [str(id) for id in range(len(chunks))]
        
        return chunk_ids, chunks



if __name__ ==  "__main__":
    token_factory = PDFChunkGenerator(pdf_path="data/microsoft-annual-report.pdf")
    token_chunks: list[str] =  token_factory.chunks
    
    print(len(token_chunks))








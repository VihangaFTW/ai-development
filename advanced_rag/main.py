from chroma import ChromaDb
from pdf_processor import PDFChunkGenerator
from query import llm_response_generator
from util import word_wrap

def main() -> None:
    db = ChromaDb()
    
    collection_name = "microsoft-collection-v3"
    collection = db.create_collection(collection_name)
    
    pdf_processor = PDFChunkGenerator('data/microsoft-annual-report.pdf')
    
    db.add_chunks(pdf_processor.get_chunk_ids(), pdf_processor.get_chunks(), collection_name)
    
    og_query = "what was Microsoft's total profit for the year 2023 and how does it compare to the previous year?"
    
    hypothetical_answer = llm_response_generator(og_query)
    
    concat_query = f"{og_query}\n{hypothetical_answer}"
    
    results = db.query_documents(question=concat_query,collection_name=collection_name, n_results=5)
    
    for result in results:  # type: ignore
        print(result)

if __name__ == "__main__":
    main()

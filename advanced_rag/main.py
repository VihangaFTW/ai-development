from chroma import ChromaDb
from pdf_processor import PDFChunkGenerator



def main() -> None:
    db = ChromaDb()
    
    collection_name = "microsoft-collection-v3"
    collection = db.create_collection(collection_name)
    
    pdf_processor = PDFChunkGenerator('data/microsoft-annual-report.pdf')
    
    db.add_chunks(pdf_processor.get_chunk_ids(), pdf_processor.get_chunks(), collection_name)
    
    query = 'what is the total revenue for the year?'
    
    results: list[str] | None = db.query_documents(query, collection_name)
    
    print(results)
    
    
    
    




if __name__ == "__main__":
    main()

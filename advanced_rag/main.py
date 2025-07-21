from chroma import ChromaDb
from pdf_processor import PDFChunkGenerator
from response import generate_single_query_response, generate_multi_query_response, generate_response_with_context
from util import word_wrap


def run_expanded_single_query(db: ChromaDb, question: str) -> None:
    """
    Execute a RAG pipeline using the HyDE (Hypothetical Document Embeddings) technique.
    
    This function implements a single-query expansion approach where an LLM generates
    a hypothetical answer to the user's question, which is then combined with the
    original query to improve semantic retrieval. The expanded query helps bridge
    vocabulary gaps between the question and document content, leading to more
    accurate document retrieval and better final responses.
    
    RAG Pipeline Steps:
        1. Create/retrieve a ChromaDB collection for document storage
        2. Process and chunk the PDF document using PDFChunkGenerator
        3. Store document chunks in the vector database with embeddings
        4. Generate a hypothetical answer using the question (HyDE technique)
        5. Combine original question + hypothetical answer for expanded retrieval
        6. Query the vector database with the expanded query
        7. Generate a comprehensive response using retrieved context
        8. Display the formatted final response
    
    HyDE Benefits:
        - Bridges semantic gaps between question and document terminology
        - Provides richer context for vector similarity matching
        - Improves retrieval accuracy for domain-specific queries
        - Enhances performance when question and answer vocabulary differ
    
    Args:
        db (ChromaDb): Initialized ChromaDB instance for vector operations
        question (str): The user's question to be answered using RAG
        
    Returns:
        None: Prints the AI-generated response and handles display formatting
        
    Example:
        >>> db = ChromaDb()
        >>> question = "What was Microsoft's profit margin in 2023?"
        >>> run_expanded_single_query(db, question)
        # Creates collection, processes PDF, generates hypothetical answer,
        # retrieves relevant chunks, and displays comprehensive response
        
    Note:
        - Uses a dedicated collection name 'microsoft-collection-single'
        - Processes 'data/microsoft-annual-report.pdf' as the document source
        - Retrieves top 5 most relevant document chunks
        - Handles cases where no relevant documents are found
    """
    print("\n" + "🔍" + "="*118 + "🔍")
    print("🚀 ADVANCED RAG DEMO - HyDE (Hypothetical Document Embeddings) Technique")
    print("="*120)
    print(f"📝 QUESTION: {word_wrap(question, line_width=100)}")
    print("="*120)
    
    collection_name = "microsoft-collection-single"
    print("📂 Step 1/6: Setting up ChromaDB collection...")
    collection = db.create_collection(collection_name)
    print("✅ Collection created/retrieved successfully")
    
    print("\n📄 Step 2/6: Processing PDF document...")
    pdf_processor = PDFChunkGenerator('data/microsoft-annual-report.pdf')
    print("✅ PDF processed and chunked")
    
    print("\n💾 Step 3/6: Storing document chunks in vector database...")
    db.add_chunks(pdf_processor.get_chunk_ids(), pdf_processor.get_chunks(), collection_name)
    print("✅ Document chunks stored with embeddings")
    
    print("\n🤖 Step 4/6: Generating hypothetical answer (HyDE)...")
    hypothetical_answer = generate_single_query_response(question)
    print("✅ Hypothetical answer generated")
    
    print("\n🔗 Step 5/6: Combining query with hypothetical answer...")
    concat_query = f"{question}\n{hypothetical_answer}"
    print("✅ Expanded query prepared")
    
    print("\n🔎 Step 6/6: Searching vector database...")
    results = db.query_documents(question=concat_query,collection_name=collection_name, n_results=5)
    print("✅ Relevant documents retrieved")
    
    # generate an llm response with the extra context
    if results and isinstance(results, list):
        print("\n🧠 Generating AI response with retrieved context...")
        response = generate_response_with_context(question, results)
        
        print("\n" + "🎯" + "="*116 + "🎯")
        print("🤖 AI RESPONSE (HyDE Technique)")
        print("="*120)
        print(word_wrap(response, line_width=120))
        print("="*120)
        print("🎉 RAG Pipeline Complete! ✨")
        print("="*120)
    else:
        print("❌ No results retrieved from the database.")


def run_expanded_multiple_queries(db: ChromaDb, question: str) -> None:
    """
    Execute a RAG pipeline using the Multi-Query Expansion technique.
    
    This function implements a comprehensive retrieval approach where an LLM generates
    multiple related subqueries from the original question. All queries are processed
    simultaneously by ChromaDB for batch retrieval, capturing different aspects and
    perspectives of the user's question. This technique provides more robust and
    comprehensive document retrieval compared to single-query approaches.
    
    RAG Pipeline Steps:
        1. Create/retrieve a ChromaDB collection for document storage
        2. Process and chunk the PDF document using PDFChunkGenerator  
        3. Store document chunks in the vector database with embeddings
        4. Generate multiple related subqueries from the original question
        5. Combine original question + subqueries into a query list
        6. Perform batch retrieval using all queries simultaneously
        7. Apply deduplication to remove duplicate document chunks
        8. Generate a comprehensive response using retrieved context
        9. Display the formatted final response
    
    Multi-Query Benefits:
        - Captures different aspects and perspectives of the original question
        - Overcomes limitations of single-query semantic search
        - Increases recall by finding documents matching related concepts
        - Reduces risk of missing relevant information due to query specificity
        - Leverages ChromaDB's efficient batch processing capabilities
        - Provides more comprehensive context for response generation
    
    Args:
        db (ChromaDb): Initialized ChromaDB instance for vector operations
        question (str): The user's question to be answered using RAG
        
    Returns:
        None: Prints the AI-generated response and handles display formatting
        
    Example:
        >>> db = ChromaDb()
        >>> question = "What factors contributed to Microsoft's revenue growth?"
        >>> run_expanded_multiple_queries(db, question)
        # Creates collection, processes PDF, generates multiple subqueries,
        # performs batch retrieval with deduplication, displays comprehensive response
        
    Note:
        - Uses a dedicated collection name 'microsoft-collection-multiple'
        - Processes 'data/microsoft-annual-report.pdf' as the document source
        - Retrieves top 5 most relevant document chunks per query
        - Automatically deduplicates results to avoid redundant context
        - Handles cases where no relevant documents are found
    """
    print("\n\n" + "🔍" + "="*118 + "🔍")
    print("🚀 ADVANCED RAG DEMO - Multi-Query Expansion Technique")
    print("="*120)
    print(f"📝 QUESTION: {word_wrap(question, line_width=100)}")
    print("="*120)
    
    collection_name = 'microsoft-collection-multiple'
    print("📂 Step 1/7: Setting up ChromaDB collection...")
    collection = db.create_collection(collection_name)
    print("✅ Collection created/retrieved successfully")
    
    print("\n📄 Step 2/7: Processing PDF document...")
    pdf_processor = PDFChunkGenerator('data/microsoft-annual-report.pdf')
    print("✅ PDF processed and chunked")
    
    print("\n💾 Step 3/7: Storing document chunks in vector database...")
    db.add_chunks(pdf_processor.get_chunk_ids(), pdf_processor.get_chunks(), collection_name)
    print("✅ Document chunks stored with embeddings")
    
    print("\n🤖 Step 4/7: Generating multiple related subqueries...")
    augmented_queries: list[str] = generate_multi_query_response(question)
    print(f"✅ Generated {len(augmented_queries)} related queries")
    
    print("\n🔗 Step 5/7: Combining all queries for batch processing...")
    concat_queries: list[str] = [question] + augmented_queries
    print(f"✅ Prepared {len(concat_queries)} total queries for search")
    
    print("\n🔎 Step 6/7: Performing batch search with deduplication...")
    results = db.query_documents(question=concat_queries,collection_name=collection_name, n_results=5)
    print("✅ Relevant documents retrieved and deduplicated")
    
    print("\n🧠 Step 7/7: Generating AI response with comprehensive context...")
    # generate an llm response with the extra context
    if results and isinstance(results, list):
        response = generate_response_with_context(question, results)
        
        print("\n" + "🎯" + "="*116 + "🎯")
        print("🤖 AI RESPONSE (Multi-Query Technique)")
        print("="*120)
        print(word_wrap(response, line_width=120))
        print("="*120)
        print("🎉 RAG Pipeline Complete! ✨")
        print("="*120)
    else:
        print("❌ No results retrieved from the database.")




def main() -> None:
    print("🌟" + "="*118 + "🌟")
    print("                    🎯 ADVANCED RAG TECHNIQUES COMPARISON DEMO 🎯")
    print("                         Powered by ChromaDB + OpenAI")
    print("="*120)
    print("📖 Document: Microsoft Annual Report 2023")
    print("🧩 Techniques: HyDE vs Multi-Query Expansion")
    print("="*120)
    
    db = ChromaDb()
    question = "What details can you provide about the factors that led to revenue growth?"
    
    # Run HyDE technique
    run_expanded_single_query(db, question)
    
    # Add separation between techniques
    print("\n" + "⚡" + "="*118 + "⚡")
    print("                          🔄 SWITCHING TO NEXT TECHNIQUE 🔄")
    print("="*120)
    
    # Run Multi-Query technique
    run_expanded_multiple_queries(db, question)
    
    # Final summary
    print("\n" + "🏆" + "="*118 + "🏆")
    print("                            ✨ DEMO COMPLETE ✨")
    print("                     Both RAG techniques successfully demonstrated!")
    print("="*120)

if __name__ == "__main__":
    main()

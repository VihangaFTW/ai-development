from util import load_and_get_key
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


api_key = load_and_get_key()
client = OpenAI(api_key=api_key)


def generate_single_query_response(query: str, model: str = "gpt-4.1-nano") -> str:
    """
    Generate a hypothetical answer for query expansion in RAG systems.

    This function implements the "Hypothetical Document Embeddings (HyDE)" technique,
    a query expansion method where an LLM generates a hypothetical answer to the user's
    question. This hypothetical answer is then combined with the original query to
    create a more comprehensive search query for the vector database. The technique
    improves retrieval accuracy by providing semantic context that may not be present
    in the original question alone.

    RAG Query Expansion Benefits:
        - Bridges vocabulary gaps between question and documents
        - Provides semantic context for better embedding similarity
        - Improves retrieval of relevant passages that use different terminology
        - Enhances search quality for complex or domain-specific queries

    Args:
        query (str): The original user question to generate a hypothetical answer for
        model (str, optional): The OpenAI model to use for generation. Defaults to 'gpt-4.1-nano'

    Returns:
        str: A hypothetical answer that can be combined with the original query

    Raises:
        ValueError: If the OpenAI API returns None content

    Example:
        >>> query = "What was Microsoft's profit in 2023?"
        >>> hypothetical = generate_single_query_response(query)
        >>> expanded_query = f"{query}\n{hypothetical}"
        >>> # Use expanded_query for vector database search

    Note:
        This function is typically used in conjunction with vector database queries
        where the expanded query (original + hypothetical answer) provides better
        semantic matching for document retrieval.
    """
    system_prompt = """You are a helpful expert financial research assistant. 
    Provide an example answer to the given question, that might be found in a document like an annual report."""

    messages: list[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    ] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(model=model, messages=messages)

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI API returned None content")

    return content


def generate_multi_query_response(query: str, model: str = "gpt-4.1-nano") -> list[str]:
    """
    Generate multiple related subqueries for comprehensive RAG retrieval.

    This function implements the "Multi-Query Expansion" technique, a RAG optimization
    method where an LLM generates multiple related questions from a single user query.
    The entire list of subqueries is passed to ChromaDB for batch processing, leveraging
    its ability to handle multiple queries simultaneously. This technique provides more
    comprehensive and accurate retrieval than a single query alone by addressing
    semantic gaps and capturing different perspectives of the original question.

    RAG Multi-Query Benefits:
        - Captures different aspects and perspectives of the original question
        - Overcomes semantic search limitations with varied query formulations
        - Increases recall by finding documents that match related concepts
        - Reduces the risk of missing relevant information due to query specificity
        - Improves robustness against query formulation variations
        - Leverages ChromaDB's batch processing for efficient multi-query search

    Args:
        query (str): The original user question to expand into multiple subqueries
        model (str, optional): The OpenAI model to use for generation. Defaults to 'gpt-4.1-nano'

    Returns:
        list[str]: A cleaned list of related questions that are processed together
                  by ChromaDB for comprehensive document retrieval. Empty strings
                  and whitespace are automatically removed.

    Raises:
        ValueError: If the OpenAI API returns None content

    Example:
        >>> original = "What was Microsoft's financial performance in 2023?"
        >>> subqueries = generate_multi_query_response(original)
        >>> print(subqueries)
        # Output: [
        #     "What was Microsoft's total revenue in 2023?",
        #     "How much profit did Microsoft make in 2023?",
        #     "What were Microsoft's operating expenses in 2023?",
        #     "How did Microsoft's stock performance change in 2023?",
        #     "What were Microsoft's key financial metrics for 2023?"
        # ]

        >>> # Use the entire list with ChromaDB for batch querying
        >>> results = db.query_documents(
        ...     question=subqueries,  # Pass the whole list
        ...     collection_name="financial_docs",
        ...     n_results=10
        ... )

    Usage in RAG Pipeline:
        1. Generate multiple subqueries using this function
        2. Pass the entire list to ChromaDB's query method for batch processing
        3. ChromaDB aggregates and returns results from all subqueries simultaneously
        4. Use the comprehensive results for enhanced response generation

    Note:
        - Each subquery is automatically cleaned of whitespace and empty strings
        - The generated queries are single-topic questions optimized for vector search
        - ChromaDB processes all queries in the list together for maximum efficiency
    """

    system_prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering (include \n after each question).
                """

    messages: list[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    ] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(model=model, messages=messages)

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI API returned None content")

    # convert to a list of queries by splitting by newline character
    queries: list[str] = content.split("\n")

    # remove empty strings and strip whitespace from each query
    # * if check filters out empty strings because empty lines become empty strings after splitting
    cleaned_queries = [query.strip() for query in queries if query.strip()]

    return cleaned_queries


def generate_response_with_context(
    query: str, context_chunks: list[str], model: str = "gpt-4.1-nano"
) -> str:
    """
    Generate a comprehensive response using the original query and retrieved document context.

    This function implements the final response generation step in a RAG pipeline. It takes
    the user's original question and the relevant document chunks retrieved from the vector
    database, then uses an LLM to synthesize a comprehensive, factual answer based on the
    provided context. The response is grounded in the retrieved documents to ensure accuracy
    and reduce hallucination.

    RAG Response Generation Benefits:
        - Provides answers grounded in actual document content
        - Reduces LLM hallucination by constraining responses to retrieved context
        - Combines multiple relevant passages for comprehensive answers
        - Maintains source attribution and factual accuracy
        - Handles cases where information spans multiple document chunks

    Args:
        query (str): The original user question that needs to be answered
        context_chunks (list[str]): List of relevant document chunks retrieved from vector database
        model (str, optional): The OpenAI model to use for generation. Defaults to 'gpt-4.1-nano'

    Returns:
        str: A comprehensive response that answers the query using the provided context

    Raises:
        ValueError: If the OpenAI API returns None content

    Example:
        >>> query = "What was Microsoft's revenue growth in 2023?"
        >>> chunks = [
        ...     "Microsoft's total revenue was $211.9 billion in fiscal 2023...",
        ...     "Revenue increased 7% year-over-year driven by cloud services...",
        ...     "The company saw strong growth in Azure and Office 365..."
        ... ]
        >>> response = generate_response_with_context(query, chunks)
        >>> print(response)
        # Output: "Based on the provided information, Microsoft achieved
        # total revenue of $211.9 billion in fiscal 2023, representing
        # a 7% year-over-year increase..."

    Usage in RAG Pipeline:
        1. User asks a question
        2. Generate expanded queries (optional)
        3. Retrieve relevant chunks from vector database
        4. Use this function to generate final response with context

    Note:
        - The function handles empty context gracefully
        - Response quality depends on the relevance of retrieved chunks
        - Model is instructed to be factual and cite information from context
    """

    # Handle case where no context is provided
    if not context_chunks:
        context_text = "No relevant context found."
    else:
        # Combine all context chunks into a single text block
        context_text = "\n\n".join(
            [f"Context {i + 1}: {chunk}" for i, chunk in enumerate(context_chunks)]
        )

    system_prompt = """You are a helpful financial research assistant. 
    Answer the user's question based solely on the provided context from financial documents. 
    Be factual, comprehensive, and cite specific information from the context when possible. 
    If the context doesn't contain enough information to fully answer the question, 
    clearly state what information is missing. Do not make up information not present in the context."""

    user_prompt = f"""Question: {query}

Context from retrieved documents:
{context_text}

Please provide a comprehensive answer based on the context above."""

    messages: list[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    ] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(model=model, messages=messages)

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI API returned None content")

    return content

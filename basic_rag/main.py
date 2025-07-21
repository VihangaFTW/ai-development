from embedding import DocumentEmbedder
from chroma import ChromaDb
from util import load_and_get_key
from openai import OpenAI
from openai.types.chat import ChatCompletion


def generate_rag_response(question: str, relevant_chunks: list[str]) -> None:
    """
    Generate a response to a question using RAG (Retrieval-Augmented Generation).

    This function takes a user question and relevant document chunks, then uses OpenAI's
    GPT model to generate a contextually informed answer based on the provided content.

    Args:
        question (str): The user's question to be answered.
        relevant_chunks (list[str]): List of relevant text chunks retrieved from the vector database
                                   that provide context for answering the question.

    Returns:
        None: Prints the generated answer to the console or an error message if generation fails.
    """

    api_key: str | None = load_and_get_key()

    if not api_key:
        print("Please add an OpenAI api key to the current environment")
        return

    client = OpenAI(api_key=api_key)

    context: str = "\n\n".join(relevant_chunks)
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    llm_response: ChatCompletion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    answer = llm_response.choices[0].message.content

    print(answer) if answer else print("Failed to generate an answer")


def main() -> None:
    """
    Main function that orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.

    This function performs the following steps:
    1. Loads and processes documents from the news_articles directory
    2. Creates chunks and generates embeddings for each chunk
    3. Sets up a ChromaDB vector database and stores the embedded chunks
    4. Queries the database with a predefined question
    5. Generates and displays an AI-powered response using the retrieved context

    The function demonstrates a complete end-to-end RAG workflow for question-answering
    over a collection of news articles about AI developments.

    Returns:
        None: Executes the RAG pipeline and prints results to console.
    """

    print("Starting RAG system...")
    question = "Has Slack started priotizing ai features in the app?"
    collection_name = "news"

    chunk_factory = DocumentEmbedder("./news_articles")
    chunks = chunk_factory.get_chunks()

    print("Setting up vector database...")
    vector_db = ChromaDb()
    vector_db.create_collection(
        collection_name=collection_name,
        metadata={"description": "a collection of news articles"},
    )
    vector_db.add_chunks(chunks, collection_name)
    print("Database setup complete")

    print(f"\nQuerying: {question}")
    relevant_chunks = vector_db.query_documents(question, collection_name)

    if not relevant_chunks:
        print(f'Vector db returned no results for the question "{question}".')
        return

    generate_rag_response(question, relevant_chunks)


if __name__ == "__main__":
    main()

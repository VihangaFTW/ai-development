import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import EmbeddingFunction, Embeddable
from datetime import datetime
from typing import cast

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# we will use openai's embedding llm instead of the default embedding llm provided by chromadb
openai_ef = OpenAIEmbeddingFunction(
    api_key = openai_key, model_name='text-embedding-3-small'
)

# initialize the chroma client with persistent storage
chroma_client = chromadb.PersistentClient()

collection = chroma_client.get_or_create_collection(
    name="newspapers",
    #? the cast is to fix a type checker bug. Alternative is to comment the line with "# type: ignore"
    embedding_function= cast(EmbeddingFunction[Embeddable],openai_ef),
    metadata={
        "description" : "my first chroma collection",
        "created": str(datetime.now())
    }
)

openai_client = OpenAI(api_key=openai_key)

# response = openai_client.chat.completions.create(
#     model = "gpt-4.1-nano",
#     messages= [
#         {"role": "system", "content": 'You are helping me to find the most relevant newspaper articles for my query. You will be given a query and a list of newspaper articles. You will need to find the most relevant articles for the query.'},
#         {"role": "user", "content": "What is the human life expectancy in the United States?"}
#     ]
# )

# print(response.choices[0].message.content)


def load_documents(directory: str) -> list[dict[str,str]]:
    documents: list[dict[str,str]] = []
    
    # loop through the files in given directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # read the file contents
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                documents.append({
                    "doc_name": filename,
                    "doc_content": f.read() 
                })
    
    return documents


# funtion to split the text into chunks
def chunk_generator(text: str, chunk_size: int=1000, chunk_overlap:int=20) -> list[str]:
    chunks: list[str] = []
    start: int = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start: end])
        start = end-chunk_overlap

    return chunks


def documents_to_chunks(documents: list[dict[str,str]]) -> list[dict[str, str]]:
    document_chunks: list[dict[str,str]] = []
    
    for doc in documents:
        print(f'{"="*50} Splitting document {doc["doc_name"]} into chunks {"="*50}')
        chunks: list[str] = chunk_generator(doc["doc_content"])
        for i,chunk in enumerate(chunks):
            document_chunks.append({
                "id": f'{doc["doc_name"]}_chunk{i+1}',
                "content": chunk
            })
        print(f'{"="*50} Chunks generated {"="*50}')
        
    
    return document_chunks



def get_openai_embeddings(text: str):
    









documents = load_documents("./news_articles")

print(f'Loaded {len(documents)} documents!')



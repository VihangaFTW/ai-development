

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import EmbeddingFunction, Embeddable
from datetime import datetime
from typing import cast
from embedding import DocumentEmbedder

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



# response = openai_client.chat.completions.create(
#     model = "gpt-4.1-nano",
#     messages= [
#         {"role": "system", "content": 'You are helping me to find the most relevant newspaper articles for my query. You will be given a query and a list of newspaper articles. You will need to find the most relevant articles for the query.'},
#         {"role": "user", "content": "What is the human life expectancy in the United States?"}
#     ]
# )

# print(response.choices[0].message.content)





def main() -> None:
    
    chunk_factory  = DocumentEmbedder("./news_articles")
    chunks = chunk_factory.get_chunks()



if __name__ == "__main__":
    main()



from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddable
from datetime import datetime
from typing import cast
from util import load_and_get_key


# we will use openai's embedding llm instead of the default embedding llm provided by chromadb
openai_ef = OpenAIEmbeddingFunction(
    api_key = load_and_get_key(), model_name='text-embedding-3-small'
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
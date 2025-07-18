from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from dotenv import load_dotenv
import os

load_dotenv()

# temperature = 0 to minimize hallucinations
llm = ChatOpenAI(model='gpt-4.1-nano', temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "Stock_Market_Performance_2024.pdf"


# check if the filepath exsits 
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f'PDF file not found: {pdf_path}')


pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f'PDF has been loaded and has {len(pages)} pages')
except Exception as e:
    print(f'Error loading PDF: {e}')

text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split =  text_splitter.split_documents(pages)


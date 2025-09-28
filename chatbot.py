# -----------------------------------------
# Shopify Chatbot with Pinecone + Streamlit
# -----------------------------------------

import os
import hashlib
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_pinecone")
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="langchain_pinecone"
)


# -------------------------------
# 0. Load environment variables
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------------------------
# 1. Pinecone setup
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "shopify"   # your Pinecone index
namespace = "shopify"    # namespace for docs

# -------------------------------
# 2. OpenAI embeddings
# -------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# -------------------------------
# 3. Scraper function
# -------------------------------
def scrape_website(url: str):
    """Return (page_title, text)"""
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator=" ", strip=True)
    return title, text

# -------------------------------
# 4. Ingest Shopify content
# -------------------------------
def ingest_shopify():
    urls = [
        "https://www.shopify.com/",
        "https://www.shopify.com/free-trial"
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    metadatas = []
    for url in urls:
        page_title, text = scrape_website(url)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "source": url,
                "title": page_title,
                "chunk_id": i
            })

    ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in all_chunks]

    PineconeVectorStore.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=index_name,
        namespace=namespace,
        ids=ids
    )

    return namespace

# -------------------------------
# 5. Rewrite vague queries
# -------------------------------
def rewrite_query(llm, user_query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "You are a rewrite assistant for a Shopify knowledge chatbot. "
            "Rewrite the user's question into a concise search query that will help "
            "retrieve relevant passages from our ingested Shopify pages. "
            "If the query is vague (like 'title'), expand it into a full Shopify-related question. "
            "User query: {query}\n\nRewrite:"))

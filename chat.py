import os
import hashlib
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# -------------------------------
# 0. Load environment variables
# -------------------------------
# Make sure you have a .env file with PINECONE_API_KEY and OPENAI_API_KEY
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------------------------
# 1. Initialize Pinecone client
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "shopify"  # Existing Pinecone index
namespace = "shopify"   # Namespace for organizing chunks

# -------------------------------
# 2. Initialize OpenAI embeddings
# -------------------------------
# Make sure the embedding dimension matches your Pinecone index (3072)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# -------------------------------
# 3. Scraper function
# -------------------------------
def scrape_website(url: str) -> str:
    """
    Scrape plain text from a website.
    Returns all visible text on the page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

# -------------------------------
# 4. Ingest Shopify content
# -------------------------------
def ingest_shopify():
    """
    Scrapes Shopify URLs, chunks text, and upserts embeddings into Pinecone.
    """
    urls = [
        "https://www.shopify.com/",
        "https://www.shopify.com/free-trial"
    ]

    # Split large text into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    for url in urls:
        text = scrape_website(url)
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    # Generate unique IDs for Pinecone upsert
    ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in all_chunks]

    # Upsert chunks into existing Pinecone index
    PineconeVectorStore.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        ids=ids  # ensures uniqueness
    )

    return namespace

# -------------------------------
# 5. Retriever for querying
# -------------------------------
def get_retriever(namespace: str):
    """
    Create a retriever from the existing Pinecone index for LLM querying.
    """
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 6. QA chain
# -------------------------------
def create_qa(namespace: str):
    """
    Initialize a RetrievalQA chain with ChatOpenAI and Pinecone retriever.
    """
    retriever = get_retriever(namespace)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # simple aggregation of chunks
    )

# -------------------------------
# 7. Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Shopify Chatbot", page_icon="🛍️")
    st.title("🛍️ Shopify Chatbot")

    # Only ingest content once per Streamlit session
    if "ingested" not in st.session_state:
        st.info("Ingesting Shopify pages into Pinecone... This may take a minute.")
        ingest_shopify()
        st.session_state["ingested"] = True
        st.success("Ingestion complete!")

    # Initialize QA chain
    qa = create_qa(namespace)

    # Chat interface
    user_q = st.text_input("Ask me anything about Shopify:")
    if user_q:
        # answer = qa.run(user_q)
        answer = qa.invoke({"query": user_q})["result"]

        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()

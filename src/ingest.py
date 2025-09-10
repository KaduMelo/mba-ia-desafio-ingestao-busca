import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
for k in ("PDF_PATH", "OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise ValueError(f"Please set the {k} environment variable in the .env file")
    
logging.info(f"Loading PDF from {os.getenv('PDF_PATH')}")
docs = PyPDFLoader(str(os.getenv("PDF_PATH"))).load()
logging.info(f"Loaded {len(docs)} pages from PDF")

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150, 
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
).split_documents(docs)

if not chunks:
    logging.warning("No chunks were created from the document")
    raise SystemExit(0)

logging.info(f"Created {len(chunks)} chunks from the document")

logging.info("Initializing OpenAI embeddings")
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

logging.info(f"Connecting to PostgreSQL database with collection {os.getenv('PG_VECTOR_COLLECTION_NAME')}")
store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

# Batch processing configuration
batch_size = 5
total_batches = (len(chunks) - 1) // batch_size + 1

logging.info(f"Starting batch ingestion of {len(chunks)} chunks in {total_batches} batches")

for i in range(0, len(chunks), batch_size):
    batch_docs = chunks[i:i + batch_size]
    store.add_documents(batch_docs)
    
    current_batch = i // batch_size + 1
    logging.info(f"✅ Batch {current_batch}/{total_batches} saved successfully")
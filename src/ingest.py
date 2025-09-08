# import os
# from pathlib import Path
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from langchain_postgres import PGVector

# load_dotenv()

# PDF_PATH = os.getenv("PDF_PATH")

# def ingest_pdf():
#     pass


# if __name__ == "__main__":
#     ingest_pdf()



import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("PDF_PATH", "OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise ValueError(f"Please set the {k} environment variable in the .env file")
    
docs = PyPDFLoader(str(os.getenv("PDF_PATH"))).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150, add_start_index=False
).split_documents(docs)

if not splits:
    raise SystemExit(0)

enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in splits
]

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

store.add_documents(documents=enriched, ids=ids)
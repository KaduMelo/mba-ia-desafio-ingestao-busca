import os
from typing import List
from dotenv import load_dotenv
import tiktoken

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

load_dotenv()
for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise ValueError(f"Please set the {k} environment variable in the .env file")

class SearchEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.store = PGVector(
            embeddings=self.embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )
        self.llm = OpenAI(temperature=0)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 3000  # Safe limit for context
        
    def _truncate_context(self, text: str) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            text = self.encoding.decode(tokens)
        return text
        
    def search_similar(self, query: str) -> List[Document]:
        return self.store.similarity_search(query, k=10)
    
    def get_response(self, query: str, docs: List[Document]) -> str:
        # Combine and truncate context
        context = "\n".join(doc.page_content for doc in docs)
        context = self._truncate_context(context)
        
        prompt = PromptTemplate.from_template(
            """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{query}

RESPONDA A "PERGUNTA DO USUÁRIO":"""
        )
        
        return self.llm.invoke(
            prompt.format(context=context, query=query)
        )
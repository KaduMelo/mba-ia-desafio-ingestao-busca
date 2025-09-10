import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise ValueError(f"Please set the {k} environment variable in the .env file")

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{query}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(query: str) -> str:
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
            
    docs = store.similarity_search_with_score(query, k=10)
    
    if not docs:
        return "Não tenho informações necessárias para responder sua pergunta."
        
    sorted_results = sorted(docs, key=lambda x: -x[1])
    context = "\n\n".join(doc.page_content for doc, _ in sorted_results)

    add_context = RunnableLambda(lambda summaries: {"context": context,"query": query})

    template_prompt = PromptTemplate(
    input_variables=["contex", "query"],
    template=PROMPT_TEMPLATE
    )
    
    llm_en = ChatOpenAI(model="gpt-5-mini", temperature=0)
    chain = add_context | template_prompt | llm_en | StrOutputParser()

    return chain.invoke({"query": query})
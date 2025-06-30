# reranker_and_llm.py

from qdrant_client import QdrantClient, models
from fastembed import LateInteractionTextEmbedding
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# ==== CONFIGURATION ====
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "multimodal_multivector"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

# ==== INITIALIZATION ====
client = QdrantClient(QDRANT_URL)
colbert_embedder = LateInteractionTextEmbedding(model_name=COLBERT_MODEL_NAME)
llm = Ollama(model="mistral")  # Assumes Ollama running Mistral locally

# ==== REACT PROMPT ====
prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    You are a helpful assistant. Answer the following query using the provided context.

    Query: {query}

    Context:
    {context}

    Provide a clear and concise answer.
    """
)

chain = LLMChain(llm=llm, prompt=prompt)

# ==== RERANK AND GENERATE ====
def rerank_and_generate(query_text: str, top_k: int = 5):
    dense_query = models.Document(text=query_text, model="BAAI/bge-small-en")
    colbert_query = models.Document(text=query_text, model=COLBERT_MODEL_NAME)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(query=dense_query, using="dense_text"),
        query=colbert_query,
        using="colbert_text",
        limit=top_k,
        with_payload=True
    )

    # Normalize: unpack tuple if needed
    def get_payload(item):
        if isinstance(item, tuple) and hasattr(item[0], "payload"):
            return item[0].payload
        elif hasattr(item, "payload"):
            return item.payload
        else:
            return {}

    payloads = [get_payload(p) for p in results]
    context = "\n".join(p.get("text") or p.get("description") or p.get("content") or "[No content]" for p in payloads)

    response = chain.run({"query": query_text, "context": context})
    return response, payloads
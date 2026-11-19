from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_storage")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="agents",
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    query_instruction="Please find passages that answer this question: ",
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

retriever = index.as_retriever(similarity_top_k=3)

def search(question):
    results = retriever.retrieve(question)
    print(f"\nQ: {question}")
    for r in results:
        print(f"score: {r.score:.3f}")
        print(r.node.metadata)
        print(r.node.text[:300])
        print("---")
    return results

search("What is Task Decomposition?")
search("What did the author have for breakfast?")

client.close()

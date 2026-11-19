from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_storage")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="agents",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
)

nodes = pipeline.run(documents=documents)

print(len(nodes))
print(len(nodes[0].embedding))
print(nodes[0].embedding[:5])
print(client.count(collection_name="agents").count)

from dotenv import load_dotenv

load_dotenv()

from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
    ]
)

nodes = pipeline.run(documents=documents)

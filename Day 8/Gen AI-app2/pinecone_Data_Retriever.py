import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Load API key
pinecone_api_key = os.getenv("Pine_cone_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Load the Pinecone index
index = pc.Index("gen-ai")

# Set up embedding and sparse encoders
ollama_embed = OllamaEmbeddings(model="nomic-embed-text")
bm25encoder = BM25Encoder().default()

# Create hybrid retriever with explicit text_key
retriever = PineconeHybridSearchRetriever(
    embeddings=ollama_embed,
    sparse_encoder=bm25encoder,
    index=index,
    text_key="text"  # Tells retriever to use 'text' key from metadata
)

# Define your query
query = "What is Nano-technology?"

# Retrieve results
results = retriever.invoke(query)

print(results)
# Display results
print("🔎 Retrieved Results:\n")
for i, doc in enumerate(results):
    print(f"{i+1}. Source: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content)
    print("-" * 80)

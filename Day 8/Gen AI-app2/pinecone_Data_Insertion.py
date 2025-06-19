import os 
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key = os.getenv('Pine_cone_API_KEY')
from pinecone import Pinecone

pc = Pinecone(api_key=pinecone_api_key)

from pinecone import ServerlessSpec


if "Gen-AI" not in pc.list_indexes().names():
    pc.create_index(
        name="gen-ai",
        dimension=768, 
        metric="dotproduct",  
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index("gen-ai")

print(index)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('nanotechnology_e.pdf')
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 75)
splitters = text_splitters.split_documents(data)

from langchain_ollama import OllamaEmbeddings
ollama_embed = OllamaEmbeddings(model='nomic-embed-text')

# print(splitters)

from pinecone_text.sparse import BM25Encoder

bm25encoder = BM25Encoder().default()




# Get dense embeddings from Ollama
dense_vectors = ollama_embed.embed_documents([doc.page_content for doc in splitters])

# Get sparse (BM25) embeddings
sparse_vectors = bm25encoder.encode_documents([doc.page_content for doc in splitters])

# Prepare data for upsert: combine dense + sparse + metadata
vectors = []
for i, doc in enumerate(splitters):
    vector_id = f"doc_{i}"

    # Get dense and sparse vectors
    dense_vector = dense_vectors[i]
    sparse_vector = sparse_vectors[i]

    # Prepare metadata (optional but useful for later search)
    metadata = {
        "text": doc.page_content,
        "source": doc.metadata.get("source", "unknown")
    }

    # Construct hybrid vector entry
    vectors.append({
        "id": vector_id,
        "values": dense_vector,
        "sparse_values": sparse_vector,
        "metadata": metadata
    })

# Upsert to Pinecone (in batches if needed)
index.upsert(vectors)

print("✅ All vectors inserted into Pinecone successfully.")

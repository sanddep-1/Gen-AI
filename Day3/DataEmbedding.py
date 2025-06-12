from langchain_community.document_loaders import PyPDFLoader

file_path = r'C:\Users\USER\OneDrive\Desktop\Krish-Naik\Gen AI\Day-3\LangchainComponents\Rohith Ravi Teja 18-232 Resume.pdf'
loader = PyPDFLoader(file_path)
text_data = loader.load()
print(text_data)
print(len(text_data))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 10)

splits = text_splitter.split_documents(text_data)

print(splits)
print(len(splits))
print(splits[0])
print(splits[0].page_content)


from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model = "nomic-embed-text")

sample_1 = "Greek First Letter is Alpha"
sample_2 = "Greek second Letter is Beta"

vec1 = embeddings.embed_query(sample_1)
vec2 = embeddings.embed_query(sample_2)
print(vec1)
print(vec2)

print(len(vec1))
from tqdm import tqdm

# final_vec = tqdm(embeddings.embed_documents(splits)) More Time to run


# ---------------------------------------------------------------------------------------------------

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from langchain_community.embeddings import HuggingFaceEmbeddings

hf_embed = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

vec_hf = hf_embed.embed_query(sample_1)

print(vec_hf)

print(len(vec_hf))
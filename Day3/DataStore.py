from langchain_community.document_loaders import TextLoader

loader = TextLoader('pk_speech.txt', encoding ='utf8')
    
text_data = loader.load()

    
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 30)

splitters = text_splitter.split_documents(text_data)
print(splitters)
print(len(splitters))

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

from langchain_community.embeddings import HuggingFaceEmbeddings

hf_embed = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')


from langchain_community.vectorstores import FAISS

db_fiass = FAISS.from_documents(splitters,hf_embed)


query1 = "Don’t expect patriotism"
results = db_fiass.similarity_search_with_score(query1)
print(results)



print("---------------------------------------------------------------------------------------------------------------------------")


from langchain_community.embeddings import OllamaEmbeddings

ollama_embed = OllamaEmbeddings(model = "nomic-embed-text")

from langchain_community.vectorstores import Chroma

db_chroma = Chroma.from_documents(splitters,ollama_embed, persist_directory="chroma_dir")

query2 = "“Bollywood stars didn’t"
results = db_chroma.similarity_search(query2)
print(results)


print("------------------------------------------------------------------------------------------------------------------")

retriver_faiss = db_fiass.as_retriever()
print(retriver_faiss.invoke(query1))


retriver_chroma = db_chroma.as_retriever()
print(retriver_chroma.invoke(query2))


db_fiass.save_local('FAISSDB')

db_chroma.persist()

print("✅ Both FAISS and Chroma vector stores have been saved locally.")
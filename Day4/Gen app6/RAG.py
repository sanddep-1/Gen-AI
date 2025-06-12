import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# print(os.environ['LANGCHAIN_API_KEY'] ,
# os.environ['LANGCHAIN_TRACING_V2'],
# os.environ['LANGCHAIN_PROJECT'])


from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model = 'gemma3:1b')


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    ("system","Behave Like an AI Assistant and answer my queries as on context {question}"),
    ("human", "{context}")
])


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('2024-wttc-introduction-to-ai.pdf')
data = loader.load()

# print(data)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 40)
splitter = text_splitters.split_documents(data)

# print(splitter)

from langchain_ollama.embeddings import OllamaEmbeddings

ollama_embed = OllamaEmbeddings(model = 'nomic-embed-text')

from langchain_community.vectorstores import Chroma

db_chroma = Chroma.from_documents(documents=splitter,embedding= ollama_embed, persist_directory='CHROMA-DB')
# db_chroma.persist()

# print("Save Succeful!")


retriver = db_chroma.as_retriever(search_type = 'similarity')

query = "As business leaders in Travel & Tourism"
resp = retriver.get_relevant_documents(query)

# print(resp)

from langchain.chains.combine_documents import create_stuff_documents_chain

chain = create_stuff_documents_chain(llm=model, prompt=prompt)

#RAG CHAIN

def rag_chain(query : str):
    docs = retriver.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs]) # DUe to context is string in chatprompt template
    
    answer = chain.invoke({"question" : query, "context" : docs})
    
    return answer

query = "As business leaders in Travel & Tourism, what are the key AI trends? answer in 3 sentences"

response = rag_chain(query)
print("AI :",response)
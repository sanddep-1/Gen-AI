import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACE_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# os.environ["USER_AGENT"] = "LangChainApp/1.0 (sandeep.gunana@gmail.com)"

from langchain_community.document_loaders import WebBaseLoader


loader = WebBaseLoader(web_path=("https://docs.smith.langchain.com/administration/tutorials/manage_spend"))

data = loader.load()

# print(data)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap =30)

splitters = text_splitters.split_documents(data)

# print(splitters)

from langchain_huggingface import HuggingFaceEmbeddings

hf_embed = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

from langchain_community.vectorstores import FAISS

db_fiass = FAISS.from_documents(splitters,hf_embed)

# db_fiass.save_local('FIASS')


retriver = db_fiass.as_retriever()

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    '''
    Answer all Question with context.
    <context>
    {context}
    </context>
    Question : {Question}
    '''
)


from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model = "gemma3:1b")

from langchain.chains.combine_documents import create_stuff_documents_chain


chian_docs = create_stuff_documents_chain(prompt=prompt,llm=model)




def rag_chain(query : str):
    
    docs = retriver.invoke(query)
    
    answer = chian_docs.invoke({"context":docs, "Question":query})
    return answer


query = "Optimization 1: manage data retention ?"
response = rag_chain(query)

print(response)



# ''' env should be used is  : (lang) PS C:\Users\USER\OneDrive\Desktop\Krish-Naik\Gen AI\Day-4\Gen AI - app2>'
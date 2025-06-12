import os 
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an AI assistant amswer Questions based on context only context :{context}"),
    ("user","{Question}")
])

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


import tempfile



st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="centered")

st.title("📄 PDF Q&A Assistant")
st.write("Upload any PDF document and ask questions based on its content.")



def process_folw(file):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)

    data = loader.load()
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 30)
    splitters = text_splitters.split_documents(data)
    hf_embed = HuggingFaceEmbeddings(model_name ='all-MiniLM-L6-v2')

    db_FAISS = FAISS.from_documents(splitters,hf_embed)
    db_FAISS.save_local('FAISS-DB')

    os.remove(tmp_path)
    
    return db_FAISS.as_retriever()


def rag_chain(query : str):
    context_docs = retriever.invoke(query)
    answer = chain.invoke({"Question":query,"context" : context_docs})
    return answer


user_pdf = st.file_uploader("🔽 Upload your PDF file here:", type=[".pdf"])



if user_pdf:
    retriever = process_folw(user_pdf)
    st.success("✅ PDF uploaded successfully! Ask any question about its content.")
    from langchain_groq import ChatGroq
    model = ChatGroq(model= 'llama3-8b-8192' , groq_proxy= groq_api_key)
    
    from langchain.chains.combine_documents import create_stuff_documents_chain

    chain = create_stuff_documents_chain(prompt=prompt, llm=model)
    
    user_question = st.text_input("what is in your mind ?")
    response = rag_chain(user_question)
    
    if user_question:
        with st.spinner("🤖 processing your answer....."):
            response = rag_chain(user_question)
            
            st.markdown("### 🧠 Answer:")
            st.write(response)
    else:
        st.info("Enter what is in your mind ?")

else:
    st.warning("please upload the pdf first !")
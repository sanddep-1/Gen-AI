import streamlit as st 

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')



st.set_page_config(page_title="RAG for PDF using NVIDIA", page_icon="📄", layout="wide")

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style='color:#4CAF50;'>📄 RAG-based PDF Chatbot with NVIDIA AI</h1>
        <p style='font-size:18px;'>Upload your PDFs, ask questions, and get answers powered by <strong>NVIDIA NIM + LangChain</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------- Upload Section ----------------
st.markdown("### 📚 Upload your PDF files:")
st.markdown("<small>Supports multiple files. Text will be extracted and used for intelligent Q&A.</small>", unsafe_allow_html=True)

pdf_data = st.file_uploader("Upload all Pds", type='.pdf', accept_multiple_files=True)

import fitz
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


from langchain_core.documents import Document


print("Imports working fine!")

llm = ChatNVIDIA(
    model="ai-llama3-70b",         # Replace with your valid model ID
    temperature=0.1,
    max_tokens=512,
)

def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text()
    return all_text


def splitters(data):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    documents = [Document(page_content=data)]
    return text_splitters.split_documents(documents)
    
    
from langchain_core.prompts import  ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    
    """
    You are an Ai Assistant, Please answer the user Questions 
    only based on context Only.
    <Question> {Question} </Question>
    <context> {context} </context>    
    """
)
    
from langchain.chains.combine_documents import create_stuff_documents_chain 
    
if pdf_data:
    
   with st.spinner("📖 Extracting text from uploaded PDFs..."):
    data = extract_text_from_pdfs(pdf_data)
    text_splitters = splitters(data)
    nvidia_embed = NVIDIAEmbeddings()

    with st.spinner("🧠 Creating and storing vector embeddings..."):
     faiss_DB = FAISS.from_documents(text_splitters,nvidia_embed)
     faiss_DB.save_local('FIASS_DB')
     
     retriever = faiss_DB.as_retriever()
     
     st.markdown("### 💬 Ask a Question from the PDF content:")
     
     query = st.text_input("Type your query here...")
     if query:
        with st.spinner("⚡ Thinking... Please wait."):
            chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
            
            def rag(query):
                context = retriever.invoke(query)
                response = chain.invoke({"Question" : query, "context" : context})
                return response
            
            answer = rag(query)
            st.success("✅ Answer:")
            st.markdown(f"<div style='padding: 10px; background-color: #f0f9ff; border-radius: 10px;'>{answer}</div>", unsafe_allow_html=True)
            
    

     
     
     
        


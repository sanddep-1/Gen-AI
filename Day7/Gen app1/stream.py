import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template(
    '''
    You are an AI assistant.
    Summarize the content in 
    a short and crisp format of 200 words.
    Context: {context}
    '''
)


st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.markdown(
    """
    🚀 Quickly summarize the content of any article, blog, or webpage using advanced LLMs.

    📌 Just paste a URL and hit **Summarize** – get a concise summary in seconds!
    """
)

url = st.text_input("🌐 Paste a webpage URL to summarize:")

st.sidebar.markdown("## 🔐 Authentication")
api_key = st.sidebar.text_input("enter Your Groq API Key : ", type='password')

# " we use api -> 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"


if url and api_key:
    
  with st.spinner("⏳ Processing ... Please wait..."):

    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,  # optional but useful if SSL errors occur
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"},)
    data = loader.load()
    
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    splitters = text_splitters.split_documents(data)
    
    llm = ChatGroq(groq_api_key = api_key, model='llama3-70b-8192')
    
    doc_chain = create_stuff_documents_chain(prompt=prompt,llm=llm)
    
    response = doc_chain.invoke({"context" : splitters})
    
    if st.button("📝 Summarize"):
        
        st.success("✅ Summary generated successfully!")
        st.write(response)
           
else:
   st.info("💡 Please provide both an API Key and a URL to summarize.")
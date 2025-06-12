import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os 

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="Langchain Multi-Tool Chatbot", page_icon="🦜", layout="centered")
st.title("🦜🔍 **Langchain Multi-Tool Chatbot**")
st.caption("🚀 Powered by [RAG Retriever], [Wikipedia], and [Arxiv] Tools")


st.sidebar.header("🔑 **Settings & Configuration**")

api_key = st.sidebar.text_input("Enter you API Key for Acces:", type='password')
temp_value = st.sidebar.select_slider("🎨 *Creativity Level (Temperature)*",[0.1*x for x in range(1,11)])
Token_count = st.sidebar.select_slider("📏 *Max Token Count*",[x for x in range(50,300,50)])


st.markdown("### 🤔 **Ask me anything!**")
query = st.text_input("💬 *What's on your mind?*")


if query:
 if api_key:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS

        loader = WebBaseLoader(web_path="https://python.langchain.com/docs/how_to/installation/")
        data = loader.load()
        text_splitters =RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        splitters = text_splitters.split_documents(data)
        ollama_embed = OllamaEmbeddings(model = 'nomic-embed-text')
        db_faiss = FAISS.from_documents(splitters,ollama_embed)
        # db_faiss.save_local('FAISS_DB')

        retriever = db_faiss.as_retriever()
        
        st.success("✅ **RAG Retriever Database initialized successfully!**")

        from langchain_core.tools import create_retriever_tool
        retriever_tool = create_retriever_tool(retriever=retriever, name='Langsmith Tool',
                                               description='anything related to langcahin or Langsmith Installation Info Related Tool.')
        
        from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
        from langchain_community.tools.arxiv import ArxivQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
        wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)
        
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max= 250)
        arxiv = ArxivQueryRun(arxiv_wrapper = arxiv_wrapper)
        
        wiki.description = "Tool for answering general knowledge questions, history, definitions, or popular topics."

        arxiv.description = "Tool for answering scientific or research-related questions using Arxiv papers."

        
        Tools = [retriever_tool,wiki,arxiv]
        
        from langchain import hub
        prompt = hub.pull('hwchase17/openai-functions-agent')
        
        from langchain_groq import ChatGroq
        groq_api_key = api_key
        
        model = ChatGroq(model_name = 'llama3-8b-8192', groq_api_key = groq_api_key,
                         temperature= temp_value, max_tokens= Token_count
                         )
        from langchain.agents import create_openai_tools_agent
        agent = create_openai_tools_agent(llm=model,tools=Tools,prompt=prompt)
        
        
        from langchain.agents import AgentExecutor
        executor =AgentExecutor(agent=agent, tools = Tools, verbose = True)
        
        with st.spinner("⏳ Processing your answer... Please wait!"):
              resposne = executor.invoke({"input" : query})
              st.success("🎉 **Response Generated:**")
              st.write(resposne)
        
 else:
       st.warning("⚠️ Please enter your **API key** in the sidebar to start the chat.")
else :
       st.info("💬 Enter a question above to begin chatting with the multi-tool agent!") 
       

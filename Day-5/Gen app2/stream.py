# import os 
# from dotenv import load_dotenv
# load_dotenv()
# import streamlit as st

# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# groq_api_key = os.getenv('GROQ_API_KEY')


# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant. Answer questions based only on the context provided.\nContext: {context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{question}")
# ])

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableMap
# from langchain_core.runnables import RunnableLambda


# import tempfile



# st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="centered")

# st.title("📄 PDF Q&A Assistant")
# st.write("Upload any PDF document and ask questions based on its content.")



# def process_folw(file):
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(file.read())
#         tmp_path = tmp_file.name
#     loader = PyPDFLoader(tmp_path)

#     data = loader.load()
#     text_splitters = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 30)
#     splitters = text_splitters.split_documents(data)
#     hf_embed = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
#     db_fiass = FAISS.from_documents(splitters,hf_embed)
#     db_fiass.save_local('FAISS_DB_QA2')

#     os.remove(tmp_path)
    
#     return db_fiass.as_retriever()




# user_pdf = st.file_uploader("🔽 Upload your PDF file here:", type=[".pdf"])



# if user_pdf:
#     retriever = process_folw(user_pdf)
#     st.success("✅ PDF uploaded successfully! Ask any question about its content.")
#     from langchain_groq import ChatGroq
#     model = ChatGroq(model_name = 'llama3-8b-8192' , groq_api_key= groq_api_key)
    
#     store = {}
#     def get_session_history(session_id)-> BaseChatMessageHistory:
#         if session_id not in store:
#             store[session_id] = ChatMessageHistory()
        
#         return store[session_id]

#     config = {"configurable":{"session_id":"chat1"}}

#     # chain = create_stuff_documents_chain(llm=model,prompt=prompt)

#     rag_chain = RunnableMap({
#         "context": lambda x: retriever.invoke(x["question"]),
#         "chat_history": lambda x: x.get("chat_history", []),
#         "question": lambda x: x["question"]
#     }) | prompt | model | StrOutputParser()

    
#     user_question = st.text_input("what is in your mind ?")
    
#     if user_question:
#         with st.spinner("🤖 processing your answer....."):
            
#             final_chain = RunnableWithMessageHistory(
#                 rag_chain,
#                 get_session_history,
#                 input_messages_key="question",
#                 history_messages_key="chat_history"
#             )

#             response = final_chain.invoke({"question": user_question}, config=config)
            
#             st.markdown("### 🧠 Answer:")
#             st.write(response)
#     else:
#         st.info("Enter what is in your mind ?")

# else:
#     st.warning("please upload the pdf first !")

import streamlit as st 
from dotenv import load_dotenv
load_dotenv()
import os


# Load Hugging Face API key
hf_api_key = os.getenv('HF_TOKEN')

# Import necessary LangChain components
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Set Streamlit UI
st.set_page_config(page_title="🤖 HuggingFace ChatBot", layout="centered")
st.title("🤖 ChatBot with HuggingFace")
st.caption("Talk to an AI powered by HuggingFace and Mistral-7B")

# Get user input
user_input = st.text_input("💬 Ask something:")

# Session management
if 'store' not in st.session_state:
    st.session_state.store = {}

if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# Set up the model and prompt only once
@st.cache_resource
def setup_model():
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    model = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_api_key,
        temperature=0.8,
        task="text-generation"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Provide helpful answers. Respond to follow-ups too."),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | model | StrOutputParser()
    return chain

# Get chat history for the current session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Process input
if user_input:
    with st.spinner("🤔 Thinking..."):

        chain = setup_model()

        chain_with_msg_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="messages"
        )

        chat_history = get_session_history(session_id)
        chat_history.add_user_message(user_input)

        config = {"configurable": {"session_id": session_id}}

        # Get response
        response = chain_with_msg_history.invoke(
            {"messages": chat_history.messages},
            config=config
        )

        # Save AI response to history
        chat_history.add_ai_message(response)

        # Show response
        st.markdown("### 🤖 Answer:")
        st.write(response)

# Optional: Display chat history
with st.expander("🕘 Chat History"):
    chat_history = get_session_history(session_id)
    for msg in chat_history.messages:
        role = "🧑 You" if msg.type == "human" else "🤖 AI"
        st.markdown(f"**{role}:** {msg.content}")

# Optional: Reset button
if st.button("🔄 Clear Conversation"):
    st.session_state.store[session_id] = ChatMessageHistory()
    st.success("Chat history cleared.")

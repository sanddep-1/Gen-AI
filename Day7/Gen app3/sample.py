import streamlit as st 
from dotenv import load_dotenv
load_dotenv()
import os 

# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

hf_api_key = os.getenv('HF_TOKEN')

from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

st.title("ChatBot with HuggingFace")

input = st.text_input("Query ?")

if not input:
    st.info("What is in your Mind")
else:
   with st.spinner("Processing your Query...."):
       
       from langchain_huggingface import HuggingFaceEndpoint
       
       repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
       model = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=hf_api_key,
                    temperature = 0.8, task="text-generation")
       
       from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
       
       prompt = ChatPromptTemplate.from_messages([
           ("system","You are an AI assistant. Provide answer to the user's question. Do not answer any follow-up questions unless explicitly asked."),

           ("human", "{query}")
       ])
       
       from langchain_core.output_parsers import StrOutputParser
       
       chain = prompt | model | StrOutputParser()
       
       response = chain.invoke({"query":input})
       
       st.subheader("Answer :")
       st.write(response)
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACE_V2'] = 'True'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')


print(os.getenv("LANGCHAIN_PROJECT"))
print(os.getenv("LANGCHAIN_API_KEY"))


from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model = "gemma3:1b")


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are an AI Assistant please Translate Question in a Language of {language}, Without any explaination"),
    ("human" , "{input}")
])  

chain = prompt | model
response = chain.invoke({"input":"How are you ?", "language": "French"})
print(response)


from langchain_core.output_parsers import  StrOutputParser

output_parser = StrOutputParser()

final_chain = prompt| model | output_parser

final_response  = final_chain.invoke({"input":"How are you ?", "language": "Hindi"})
print(final_response)

import streamlit as st 

st.write("Simple language Translator AppLicaton")

input_query = st.text_input("What sentence you want to Translate")
language_query = st.text_input("Language you want to Translate")

st_response = final_chain.invoke({"input":input_query, "language": language_query})

st.write(st_response)


from dotenv import load_dotenv
load_dotenv()

import os 

hf_token = os.getenv('HF_TOKEN')

from langchain_huggingface import HuggingFaceEndpoint

repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'

llm = HuggingFaceEndpoint(repo_id=repo_id, 
    task="text-generation", temperature=0.7,  huggingfacehub_api_token=hf_token, max_new_tokens=3000)


print(llm.invoke('What is ML ?'))


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an AI assistant answer thye following question"),
    ("human", "question : {input}")
    
])

from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"What is Gen AI"}))

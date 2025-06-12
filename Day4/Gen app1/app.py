import os
from dotenv import load_dotenv
load_dotenv()


from langchain_ollama import OllamaLLM

model = OllamaLLM(model ="gemma3:1b")

# print(model)

query = "What is Transformer Architecture"

# print(model.invoke(query))

from langchain_core.prompts import ChatPromptTemplate

sys_msg = "You are an Electrtical Engineer Answer the questions Accordingly."

prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    ("human", "{input}")
])

chain_1 = prompt | model

# response_1 = chain_1.invoke({"input" : "What is a Attention All you Need Mechanism ? Answer in 2 sentences"})

# print(response_1)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain_2 = prompt | model | output_parser


response_2 = chain_2.invoke({"input" : "What is a Transformer? Answer in 4 sentences"})

print(response_2)



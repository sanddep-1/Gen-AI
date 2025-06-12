import os 
from dotenv import load_dotenv
load_dotenv()



os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# print(os.environ['LANGCHAIN_API_KEY'])
# print(os.environ['LANGCHAIN_PROJECT'])
# print(os.environ['GROQ_API__KEY'])
# print(os.environ['LANGCHAIN_TRACING_V2'])

from langchain_groq import ChatGroq

model = ChatGroq(model='llama3-8b-8192')

print(model)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    
    ("system", "you are an AI Assistant act accordingly for language translation to {language}"),
    ("human", "{input}")
])


from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | model | output_parser

resp = chain.invoke({"input" : "How are you, Doing Good?","language":"Hindi"})

print(resp)


from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title = 'LANGCHAIN_SERVER',
    version ='1.0',
    description = 'Gen AI App Using LECEL'    
)



add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)



''' Not worked as Pydantic and Fast api version mismatch'''


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clean up LangChain tracing variables
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

# Use ChatOllama instead of OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Build the chat model (Use a model that supports chat history well)
model = ChatOllama(model="gemma3:1b")  # Try llama3, mistral, or phi3 if gemma3 fails

# Define a prompt that uses history
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Combine prompt and model
chain = prompt | model | StrOutputParser()

# Session memory store
store = {}

# Define memory getter
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap with memory
final_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Session config
config = {"configurable": {"session_id": "chat1"}}

# First message
response1 = final_chain.invoke({"input": "Hey My Name is Saandeep"}, config=config)
print("AI:", response1)

print("---------------------------------")

# Second message
response2 = final_chain.invoke({"input": "My Favourate Color is Black"}, config=config)
print("AI:", response2)

response3 = final_chain.invoke({"input" : "What is my name and my fav color ?"}, config=config)

print("AI:",response3)

config2 = {"configurable" :{"session_id":"chat2"}}

resp = final_chain.invoke({"input":"Did you remeber my name or fav color ?"},config=config2)
print(resp)


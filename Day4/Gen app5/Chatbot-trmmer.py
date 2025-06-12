import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# print(os.environ['LANGCHAIN_PROJECT'])
# print(os.environ['LANGCHAIN_API_KEY'])
# print(os.environ['LANGCHAIN_TRACING_V2'])

from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model = 'gemma3:1b')


from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
    
])


chain_1 = prompt | model

store = {}


from langchain_core.messages import trim_messages

MAX_MESSAGES_COUNT = 4
MAX_TOKENS = 50

def get_seesion_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    else:
       if len(store[session_id].messages) > MAX_MESSAGES_COUNT:
            store[session_id].messages[:] = store[session_id].messages[-MAX_MESSAGES_COUNT:]
    return store[session_id]

with_message_history = RunnableWithMessageHistory(chain_1 , 
                                                  get_seesion_history,
                                                  input_messages_key= "input",
                                                  history_messages_key= "history"
                                                  )

config = {"configurable":{"session_id":"chat1"}}

resp1 = with_message_history.invoke({"input":"Hi My Name is Saandeep, My Hobby is to Read Techie Books, My usFav color is Black !"},config=config)
print(resp1)

print("-------------------------------------")
resp3 = with_message_history.invoke({"input":"What is 2 +2"},config=config)
print(resp3)
resp4 = with_message_history.invoke({"input":"Sai Reads Books in his free time"},config=config)
print(resp4)

print("--------------------------")

resp5 = with_message_history.invoke({"input":"what is The first math problem i asked you ?"},config=config)
print(resp5)
print("----------------------")
resp6 = with_message_history.invoke({"input":"what is my Hobby ?"},config=config)
print(resp6)

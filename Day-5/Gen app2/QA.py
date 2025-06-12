import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv('GROQ_API_KEY')



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader(r'C:\Users\USER\OneDrive\Desktop\Krish-Naik\Gen AI\Day-3\LangchainComponents\Rohith Ravi Teja 18-232 Resume.pdf')
data = loader.load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 30)
splitters = text_splitters.split_documents(data)
hf_embed = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
db_fiass = FAISS.from_documents(splitters,hf_embed)
db_fiass.save_local('FAISS_DB_QA2')

retriver = db_fiass.as_retriever()



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
model = ChatGroq(model_name = 'llama3-8b-8192', groq_api_key = groq_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Answer questions based only on the context provided.\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])


from langchain.chains.combine_documents import create_stuff_documents_chain
chain = create_stuff_documents_chain(llm=model,prompt=prompt)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableLambda


store = {}
def get_session_history(session_id)-> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    return store[session_id]

config = {"configurable":{"session_id":"chat1"}}



rag_chain = RunnableMap({
    "context": RunnableLambda(lambda x: retriver.invoke(x["question"])),
    "question": RunnableLambda(lambda x: x["question"]),
    "chat_history": RunnableLambda(lambda x: x.get("chat_history", []))
}) | chain

    
final_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

query = "what is Rohit age ?"

response = final_chain.invoke({"question": query},config=config)

print(response)

print("---------------------")

query2 = "what is Rohit Phone Number?"

response2 = final_chain.invoke({"question": query2},config=config)

print(response2)

print("----------------------")

query3 = "what questions i asked you Rohit  before ?"

response3 = final_chain.invoke({"question": query3},config=config)

print(response3)
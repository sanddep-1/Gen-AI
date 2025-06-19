import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages : Annotated[list, add_messages]
    
from langgraph.graph import StateGraph, START, END
graph_builder = StateGraph(State)

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)


tool = [wiki]

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader('cheguvara.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 0)
splitters = text_splitter.split_documents(data)
ollama_embed = OllamaEmbeddings(model = "nomic-embed-text")
db_fiass = FAISS.from_documents(splitters, ollama_embed)
db_fiass.save_local('FAISS_DB')

retriever = db_fiass.as_retriever()


from langchain.tools.retriever import create_retriever_tool

retriver_tool = create_retriever_tool(retriever,description = "Tools for Search realted to chegugvera Speech", name = "Rag Tool")


Tools = [wiki, retriver_tool]

llm_with_tools = llm.bind_tools(Tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Tool Node Logic
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
tool_node = ToolNode(tools=Tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

from IPython.display import Image, display

mermaid_code = graph.get_graph().draw_mermaid()

# Print it to console
print(mermaid_code)

# Save to .mmd file
with open("chatbot_graph.mmd", "w") as f:
    f.write(mermaid_code)

print("Mermaid diagram saved to chatbot_graph.mmd")

# ---------------------------
# Run the Chat
# ---------------------------
user_input = input("Enter your query -> ")


from langchain_core.messages import AIMessage  

events = graph.stream(
    {"messages": [("user", user_input)]}, stream_mode="values"
)

print("\n🤖 Assistant response:")
for event in events:
    event["messages"][-1].pretty_print()
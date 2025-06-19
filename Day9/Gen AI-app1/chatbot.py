import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key = groq_api_key, model_name = 'llama3-8b-8192')

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END 
# from langgraph.graph.message import add_message 

class State(TypedDict):

    messages : list
    
graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    print("🧠 LLM Raw Response:", response)
    return {"messages": state["messages"] + [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

print(graph)

from IPython.display import Image, display

mermaid_code = graph.get_graph().draw_mermaid()

# Print it to console
print(mermaid_code)

# Save to .mmd file
with open("chatbot_graph.mmd", "w") as f:
    f.write(mermaid_code)

print("Mermaid diagram saved to chatbot_graph.mmd")

# To view from VS code  https://mermaid.live go link and paste the mmd code 


user_input = str(input("enter you Query ->"))

initial_state = {
    "messages": [{"role": "user", "content": user_input}]
}



from langchain_core.messages import AIMessage  

print("\n🤖 Assistant response:")
for state in graph.stream(initial_state):
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            print(last_message.content)
        else:
            print(str(last_message))
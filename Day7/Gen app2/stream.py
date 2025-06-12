from dotenv import load_dotenv
import os 
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

Tools = []
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an AI Math Assistant. You answer Math-related problems. If user asks non-Math problems, reply: 'I am a Math Assistant; please ask Math-related questions only.'"),
    ("human","{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
    
])


st.title("Math solver")

api_key  = st.sidebar.text_input("enter Groq API Key", type='password')
query = st.text_area("What is you Math Query ?")

if api_key :
  
  if query :

    Tools = []

    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
    wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper, name='Wikipedia Tool',
                            description= "Tool is to search the Math problem via Wikipedia for the solution")

    Tools.append(wiki)
    
    from langchain_groq import ChatGroq
    llm = ChatGroq(model='llama3-70b-8192', groq_api_key = api_key)
    
    from langchain.chains import LLMMathChain

    math_chain = LLMMathChain(llm = llm)

    from langchain.chains import LLMChain
    
    from langchain.agents import Tool

    calculator = Tool(name='Calculator', description='Tool for Math related query only input in math expression',
                    func=math_chain.run)

    Tools.append(calculator)
    
    chain = LLMChain(prompt = prompt,llm = llm)

    reasoning_tool = Tool(name = 'Reasoning Tool', func=chain.run,
                        description='Tool for Assuming Logic Based Question')
                            
    Tools.append(reasoning_tool)
    
    from langchain.agents import create_openai_tools_agent
    from langchain.agents import AgentExecutor

    agent = create_openai_tools_agent(llm=llm,tools=Tools, prompt = prompt)

    executor = AgentExecutor(agent=agent, tools=Tools, verbose=True)
    
    with st.spinner("Processing you Math Intution:"):
        response = executor.invoke({"input":query})

        st.write(response["output"])
    
  else :
      st.info("What is in you Mind ?")
else:
    st.info("please Enter API Key in side bar")




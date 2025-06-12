import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


os.environ["USER_AGENT"] = "my-langchain-app/0.1"


groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq
model = ChatGroq(model_name = 'llama3-8b-8192', groq_api_key = groq_api_key)

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template(
#     """You are a helpful assistant who uses tools when needed.

# {input}

# {agent_scratchpad}"""
# )


from langchain import hub

# Example of a real, publicly available prompt repo
prompt = hub.pull('hwchase17/openai-functions-agent')  # note: check exact repo name

# print(prompt)



from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

loader = WebBaseLoader(web_path="https://python.langchain.com/docs/how_to/installation/")
data = loader.load()
text_splitters =RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
splitters = text_splitters.split_documents(data)
ollama_embed = OllamaEmbeddings(model = 'nomic-embed-text')
db_faiss = FAISS.from_documents(splitters,ollama_embed)
db_faiss.save_local('FAISS_DB')

retriever = db_faiss.as_retriever()


tools = []

from langchain_core.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(retriever=retriever, name="Langsmith Installation Guide."
                                    ,description="Search any Info related to langsmith Installation form here")

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

wiki_api = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper = wiki_api)

arxiv_api = ArxivAPIWrapper(top_k_results=1,max_query_length=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

Tools = [retriever_tool, wiki,arxiv]


from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(tools=Tools, prompt=prompt, llm=model)


from langchain.agents import AgentExecutor

executor = AgentExecutor(agent=agent,tools=Tools,  verbose=True)

query1 = "What is Machine Learning ?"
response1 = executor.invoke({"input":query1 })

print(response1)

print("--------------------------------------------")

query2 = "What is Langsmith Ecosystem ?"
response2 = executor.invoke({"input":query2 })

print(response2)

print("--------------------------------------------")


query3 = "What is This paper about 1706.03762"
response3 = executor.invoke({"input":query3 })

print(response3)


import time 

queries = [
    "What is Machine Learning?",
    "How do I install Langsmith?",
    "Tell me about Python programming language.",
]

for query in queries:
    print(f"\n=== Query: {query} ===\n")
    try:
        response = executor.invoke({"input": query})
        print("Response:\n", response)
    except Exception as e:
        print("Error during query execution:", e)
        print("Sleeping 10 seconds before retrying...")
        time.sleep(10)


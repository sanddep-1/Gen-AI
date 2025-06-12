import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

print(os.environ['LANGCHAIN_API_KEY'],",",os.environ['LANGCHAIN_TRACING_V2'],",",os.environ['LANGCHAIN_PROJECT'] )


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    
    ("system", "You are an AI ENgineer Answer the Questions accordingly"),
    ("user","{Question}")
])


from langchain_ollama.llms import OllamaLLM

def process(model,temperature,tokens,input_text):
    llm = OllamaLLM(model = model,temperature=temperature,max_tokens = tokens)

    chain = prompt | llm
    
    answer = chain.invoke({"Question":input_text})
    return answer




import streamlit as st

st.title("AI-Powered Knowledge Assistant Using Ollama")
input_text = st.text_input("Ask me anything 💬")

model = st.sidebar.selectbox("🔧 Choose a Language Model:",["-- Select a model --",'gemma3','gemma3:1b'])

temperature =st.sidebar.select_slider("🎛️ Response Creativity (Temperature):",options=[x*0.1 for x in range(1,11)],
                        help="Lower values make answers more focused and deterministic; higher values increase creativity.")

tokens = st.sidebar.select_slider("📏 Maximum Response Length (Tokens):",[x for x in range(50,300,50)])

# print(temperature,tokens, model)

if model != "-- Select a model --" and input_text:
    with st.spinner("Generating answer... Please wait."):
        response = process(model,temperature,tokens,input_text)
        st.subheader("📘 Answer:")
        st.write(response)
elif input_text:
    st.warning("⚠️ Please select a language model from the sidebar.")
else:
    st.info("💡 Type your question above to get started.")
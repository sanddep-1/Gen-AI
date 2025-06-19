from dotenv import load_dotenv
import os 
import streamlit as st

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')



from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    '''
    You are a Coding Assistant 
    provide the Scripts as per the Questions provided by user,
    if they ask any other questions say them like I am a coding assitant 
    developed by Saandeep please ask only coding related quetsions. 
    Question : {Question}
    Script :
    '''
)


st.set_page_config(page_title="CodeME - Codellama Assistant", layout="centered")
st.title("💻 CodeME - Codellama Coding Assistant")

query = st.text_input("🧠 What coding-related question do you have?")

final_prompt = prompt.format(Question=query)

if not query:
    st.info("👆 Enter a coding question to get started!")
else:
  with st.spinner("🤖 Codellama is generating your code... Please wait!"):
    import requests
    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json={
        "model": "codeme",  # Your custom model
        "prompt": final_prompt,
        "stream": False
    })

    if response.status_code == 200:
        st.write("Codeme :\n")
        st.write(response.json()["response"])
    else:
        st.write("Error:", response.status_code)
        st.write(response.text)
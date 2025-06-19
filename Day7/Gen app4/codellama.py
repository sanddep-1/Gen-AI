from dotenv import load_dotenv
import os 

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGcHAIN_PRoject')



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

query = str(input("What is Coding related Query -->"))

final_prompt = prompt.format(Question=query)

import requests
url = "http://localhost:11434/api/generate"

response = requests.post(url, json={
    "model": "codeme",  # Your custom model
    "prompt": final_prompt,
    "stream": False
})

if response.status_code == 200:
    print("Codeme :\n")
    print(response.json()["response"])
else:
    print("Error:", response.status_code)
    print(response.text)
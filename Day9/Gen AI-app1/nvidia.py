import os
import json
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv('NVIDIA_API_KEY')

# Validate API key
if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env file")

# Stream flag
stream = True

# API endpoint and headers
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json"
}

# Prompt/query
query = "What is ML?"

# Payload for the POST request
payload = {
    "model": "meta/llama-4-scout-17b-16e-instruct",
    "messages": [{"role": "user", "content": query}],
    "max_tokens": 512,
    "temperature": 1.00,
    "top_p": 1.00,
    "frequency_penalty": 0.00,
    "presence_penalty": 0.00,
    "stream": stream
}

# Make the API request
response = requests.post(invoke_url, headers=headers, json=payload)

# Parse and print streamed response
if stream:
    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data_str = decoded.replace("data: ", "")
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    delta = data_json["choices"][0]["delta"]
                    content = delta.get("content", "")
                    full_response += content
                except Exception as e:
                    print("Error parsing chunk:", e)
    print("\n🧠 Model Response:\n", full_response.strip())
else:
    print("\n🧠 Model Response:\n", response.json()["choices"][0]["message"]["content"])

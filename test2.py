import requests

url = "http://127.0.0.1:8000/query"

data = {
    "question": "What is RAG?"
}

response = requests.post(url, json=data)

print(response.json())
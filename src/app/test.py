import json
import requests

response = requests.get("http://localhost:5000/")
print(json.loads(response.text))
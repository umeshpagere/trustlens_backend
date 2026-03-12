import requests
import json

url = "http://127.0.0.1:5000/api/analyze"
payload = {"text": "Amitabh Bachchan purchased land in Ayodhya."}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error testing API: {e}")

import requests
from datetime import datetime

# -------------------------
# Update these values
# -------------------------
API_URL = "http://127.0.0.1:8000/violations/"  # Your FastAPI endpoint
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZW1haWwiOiJhZG1pbkBnbWFpbC5jb20iLCJleHAiOjE3NjA2MzQxNDl9.qKhOvuDkzb4zyvyML3I6t4pzX-86w5HUmRzxNYLJ2fI"     # If your API uses auth

# -------------------------
# Example violation payload
# -------------------------
payload = {
    "violation_types": "No Helmet",
    "worker_id": 1,              # ID of an existing worker
    "worker_code": "1",
    "frame_ts": datetime.now().isoformat()
    
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"  # remove if no auth
}

response = requests.post(API_URL, json=payload, headers=headers)

if response.status_code == 200 or response.status_code == 201:
    print("Violation created successfully!")
    print(response.json())
else:
    print("Failed to create violation:", response.status_code, response.text)

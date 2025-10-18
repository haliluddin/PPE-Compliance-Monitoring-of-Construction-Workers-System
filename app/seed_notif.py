import requests
from datetime import datetime

# -------------------------
# Configuration
# -------------------------
API_URL = "http://127.0.0.1:8000/"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyIiwiZW1haWwiOiJhcHJpbEBnbWFpbC5jb20iLCJleHAiOjE3NjA3ODAzNTJ9.DpSkx3IsomC-mZo7_C70VAlCES9Vi7XcOiILOr_PaQ0"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",  
}

# -------------------------
# Helper functions
# -------------------------
def get_workers():
    """Fetch list of workers"""
    url = f"{API_URL}/workers/"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()
    print(" Failed to fetch workers:", res.text)
    return []

def get_cameras():
    """Fetch list of cameras"""
    url = f"{API_URL}/cameras/"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()
    print("Failed to fetch cameras:", res.text)
    return []

def choose_from_list(items, key_name="name"):
    """Prompt user to choose an item"""
    if not items:
        print("No items available.")
        return None
    print("\nSelect an option:")
    for i, item in enumerate(items, 1):
        print(f"{i}. {item.get(key_name)} (ID: {item.get('id')})")
    while True:
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(items):
                return items[choice - 1]
            print("Invalid choice, try again.")
        except ValueError:
            print("Enter a number.")

# -------------------------
# Main logic
# -------------------------
def main():
    print("=== Create a New Violation ===")

    # 1️⃣ Choose a worker
    workers = get_workers()
    worker = choose_from_list(workers, key_name="fullName")
    if not worker:
        print("No worker selected. Exiting.")
        return

    # 2️⃣ Choose a camera
    cameras = get_cameras()
    camera = choose_from_list(cameras, key_name="name")
    if not camera:
        print("No camera selected. Exiting.")
        return

    # 3️⃣ Choose violation type
    print("\nViolation types:")
    types = ["No Helmet", "No Vest", "No Boots", "No Gloves"]
    for i, t in enumerate(types, 1):
        print(f"{i}. {t}")
    while True:
        try:
            t_choice = int(input("Choose violation type number: "))
            if 1 <= t_choice <= len(types):
                violation_type = types[t_choice - 1]
                break
            print("Invalid choice.")
        except ValueError:
            print("Enter a number.")

    # 4️⃣ Construct payload
    payload = {
        "violation_types": violation_type,
        "worker_id": worker["id"],
        "worker_code": worker["worker_code"],
        "camera_id": camera["id"],
        "frame_ts": datetime.now().isoformat(),
    }

    print("\n📦 Sending payload:", payload)

    # 5️⃣ Send request
    res = requests.post(f"{API_URL}/violations/", json=payload, headers=headers)
    if res.status_code in (200, 201):
        print(" Violation created successfully!")
        print(res.json())
    else:
        print("Failed to create violation:", res.status_code, res.text)

# -------------------------
# Run the script
# -------------------------
if __name__ == "__main__":
    main()

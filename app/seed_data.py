# app/seed_data.py
from app.database import SessionLocal
from app.models import User, Worker, Camera, Violation
from datetime import datetime

db = SessionLocal()

user_id = 1  # user

camera_data = [
    {"name": "Entrance Cam", "location": "Main Gate"},
    {"name": "North Site Cam", "location": "North Wing"},
]

for cam in camera_data:
    existing_cam = (
        db.query(Camera)
        .filter_by(name=cam["name"], user_id=user_id)
        .first()
    )
    if not existing_cam:
        db.add(Camera(name=cam["name"], location=cam["location"], user_id=user_id))
        print(f" Added camera: {cam['name']}")
    else:
        print(f"⚠️ Camera already exists: {cam['name']}")

db.commit()

# Fetch existing workers and cameras
workers = db.query(Worker).filter_by(user_id=user_id).all()
cameras = db.query(Camera).filter_by(user_id=user_id).all()

if not workers:
    print("No workers found for this user! Please add workers first.")
    db.close()
    exit()

print("\n Current Workers:")
for w in workers:
    print(f" - {w.id}: {w.fullName} ({w.worker_code})")

print("\n Current Cameras:")
for c in cameras:
    print(f" - {c.id}: {c.name} ({c.location})")


new_violations = [
    Violation(
        worker_id=workers[2].id,        #  worker
        camera_id=cameras[0].id,        #  camera
        user_id=user_id,
        worker_code=workers[0].worker_code,
        violation_types="No Helmet",
        frame_index=55,
        frame_ts=datetime(2025, 10, 14, 15, 30),
        inference={"detected": True},
        status="pending",
    )
]

# Add and commit
db.add_all(new_violations)
db.commit()
print("\n New violation(s) added successfully!")

db.close()

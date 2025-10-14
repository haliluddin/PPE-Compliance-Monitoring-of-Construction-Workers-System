# app/seed_cameras_and_violations.py
from database import SessionLocal
from models import User, Worker, Camera, Violation
from datetime import datetime

db = SessionLocal()

user_id = 1  # <-- adjust based  user

# Create cameras for this user
cameras = [
    Camera(name="Entrance Cam", location="Main Gate", user_id=user_id),
    Camera(name="North Site Cam", location="North Wing", user_id=user_id),
]

db.add_all(cameras)
db.commit()
print(" Cameras added!")

# Fetch workers for this user
workers = db.query(Worker).filter_by(user_id=user_id).all()

if not workers:
    print("No workers found for this user! Please add workers first.")
    db.close()
    exit()

for w in workers:
    print(f" Worker ID: {w.id}, Name: {w.fullName}")

# Fetch cameras again after commit (to get IDs)
cameras = db.query(Camera).filter_by(user_id=user_id).all()

# Simulate detection / manually add violations
test_violations = [
    Violation(
        worker_id=workers[0].id,        # Worker 1
        camera_id=cameras[0].id,        # Camera 1
        user_id=user_id,
        worker_code=workers[0].worker_code,
        violation_types="No Helmet",
        frame_index=10,
        frame_ts=datetime(2025, 10, 14, 10, 45),
        inference={"detected": True},
        status="pending",
    ),
    Violation(
        worker_id=workers[1].id,        # Worker 2
        camera_id=cameras[1].id,        # Camera 2
        user_id=user_id,
        worker_code=workers[1].worker_code,
        violation_types="No Gloves",
        frame_index=22,
        frame_ts=datetime(2025, 10, 14, 11, 15),
        inference={"detected": True},
        status="reviewed",
    ),
]

db.add_all(test_violations)
db.commit()

print("Violations added successfully!")
db.close()

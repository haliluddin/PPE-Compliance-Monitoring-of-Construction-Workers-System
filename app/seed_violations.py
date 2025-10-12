# app/seed_violations.py
from database import SessionLocal
from models import Worker, Violation
from datetime import datetime

db = SessionLocal()


worker_id = 1  

new_violations = [
    Violation(
        # job_id=None,
        # camera_id=None,
        worker_id=worker_id,
        worker_code="1",
        violation_types="No Helmet",
        frame_index=1,
        frame_ts=datetime(2025, 10, 8, 10, 30),
        snapshot=None,
        inference={"detected": False},
    ),
    Violation(
        # job_id=None,
        # camera_id=None,
        worker_id=worker_id,
        worker_code="1",
        violation_types="No Vest",
        frame_index=2,
        frame_ts=datetime(2025, 10, 9, 14, 15),
        snapshot=None,
        inference={"detected": False},
    ),
]

db.add_all(new_violations)
db.commit()
print("Violations added successfully!")
db.close()

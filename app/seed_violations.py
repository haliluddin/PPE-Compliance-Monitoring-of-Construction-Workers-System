# app/seed_violations.py
from database import SessionLocal
from models import Worker, Violation, User
from datetime import datetime

db = SessionLocal()

user_id = 1

worker_id = 6

user = db.query(User).filter_by(id=user_id).first()
worker = db.query(Worker).filter_by(id=worker_id, user_id=user_id).first()

if not user:
    print(f"❌ User with ID {user_id} not found.")
    db.close()
    exit()

if not worker:
    print(f"❌ Worker with ID {worker_id} not found.")
    db.close()
    exit()

new_violations = [
    Violation(
        worker_id=worker_id,
        user_id=user_id,
        worker_code=worker.worker_code,
        violation_types="No Gloves",
        frame_index=1,
        frame_ts=datetime(2025, 10, 8, 10, 30),
        snapshot=None,
        inference={"detected": False},
        status="Pending",
    ),
    # Violation(
    #     worker_id=worker_id,
    #     user_id=user_id,
    #     worker_code=worker.worker_code,
    #     violation_types="No Vest",
    #     frame_index=2,
    #     frame_ts=datetime(2025, 10, 9, 14, 15),
    #     snapshot=None,
    #     inference={"detected": True},
    #     status="reviewed",
    # ),
    # Violation(
    #     worker_id=worker_id,
    #     user_id=user_id,
    #     worker_code=worker.worker_code,
    #     violation_types="No Gloves",
    #     frame_index=3,
    #     frame_ts=datetime(2025, 10, 10, 9, 45),
    #     snapshot=None,
    #     inference={"detected": True},
    #     status="resolved",
    # ),
]

# Add and commit
db.add_all(new_violations)
db.commit()
print(f" {len(new_violations)} violations added successfully for User ID {user_id} and Worker ID {worker_id}.")
db.close()


# check_workers.py
# from database import SessionLocal
# from models import Worker

# db = SessionLocal()

# workers = db.query(Worker).all()
# for w in workers:
#     print(f"ID: {w.id}, Name: {w.fullName}, User ID: {w.user_id}")

# db.close()

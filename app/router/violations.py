# app/routers/violations.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Violation, Worker, User, Camera
from app.router.auth import get_current_user
from app.schemas import ViolationCreate, ViolationResponse
from app.models import Notification
from app.router.notifications_ws import connected_clients
import json


router = APIRouter(prefix="/violations", tags=["Violations"])

@router.get("/")
def get_violations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    violations = (
        db.query(Violation, Worker.fullName, Camera.name.label("camera_name"), Camera.location.label("camera_location"))
        .join(Worker, Violation.worker_id == Worker.id)
        .join(Camera, Violation.camera_id == Camera.id)
        .filter(Violation.user_id == current_user.id)  
        .all()
    )

    return [
        {
            "id": v.Violation.id,
            "violation": v.Violation.violation_types,
            "worker": v.fullName,
            "worker_code": v.Violation.worker_code,
            "camera": v.camera_name or v.camera_location,
            "frame_ts": v.Violation.frame_ts,
            "status": v.Violation.status,
        }
        for v in violations
    ]


@router.post("/")
def create_violation(
    violation: ViolationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1️⃣ Create violation record
    new_violation = Violation(
        violation_types=violation.violation_types,
        worker_id=violation.worker_id,
        frame_ts=violation.frame_ts,
        worker_code=violation.worker_code,
        user_id=current_user.id,
        status="pending",
    )
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)

    # 2️⃣ Create a corresponding notification
    message = f"New violation detected: {new_violation.violation_types} by worker code {new_violation.worker_code}"
    new_notification = Notification(
        message=message,
        user_id=current_user.id,
        violation_id=new_violation.id,
        is_read=False
    )
    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)

    # 3️⃣ Broadcast the notification via WebSocket
    notification_data = {
        "id": new_notification.id,
        "message": message,
        "is_read": new_notification.is_read,
        "created_at": str(new_notification.created_at),
        "violation_type": new_violation.violation_types,
        "worker_code": new_violation.worker_code,
    }

    # send to all connected clients of this user
    if current_user.id in connected_clients:
        for ws in connected_clients[current_user.id]:
            try:
                import asyncio
                asyncio.create_task(ws.send_text(json.dumps(notification_data)))
            except Exception as e:
                print(f"Error sending websocket notification: {e}")

    return new_violation

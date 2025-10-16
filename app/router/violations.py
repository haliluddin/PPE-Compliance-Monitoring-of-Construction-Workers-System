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
import asyncio
from app.router.notifications_ws import connected_clients, broadcast_notification

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
    # 1️⃣ Create violation
    new_violation = Violation(
        violation_types=violation.violation_types,
        worker_id=violation.worker_id,
        frame_ts=violation.frame_ts,
        worker_code=violation.worker_code,
        camera_id=violation.camera_id,
        user_id=current_user.id,
        status="pending",
    )
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)

    # 2️⃣ Create notification
    message = f"New violation detected: {new_violation.violation_types} by worker code {new_violation.worker_code}"
    new_notification = Notification(
        message=message,
        user_id=current_user.id,
        violation_id=new_violation.id,
        is_read=False,
    )
    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)

    # 3️⃣ Prepare data
    camera = db.query(Camera).filter(Camera.id == new_violation.camera_id).first()
    notification_data = {
        "id": new_notification.id,
        "message": message,
        "is_read": new_notification.is_read,
        "created_at": str(new_notification.created_at),
        "violation_type": new_violation.violation_types,
        "worker_code": new_violation.worker_code,
        "camera": camera.name if camera else "Unknown Camera",
        "camera_location": camera.location if camera else "Unknown Location",
    }

    # 4️⃣ Broadcast to WebSocket clients (live)
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast_notification(current_user.id, notification_data))
    except RuntimeError:
        # when called from a thread with no running loop
        asyncio.run(broadcast_notification(current_user.id, notification_data))

    return new_violation
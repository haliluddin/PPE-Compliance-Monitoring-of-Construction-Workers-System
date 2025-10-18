# app/routers/violations.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Violation, Worker, User, Camera, Notification
from app.router.auth import get_current_user
from app.schemas import ViolationCreate, ViolationResponse
from app.router.notifications_ws import connected_clients, broadcast_notification
import json
import asyncio

router = APIRouter(prefix="/violations", tags=["Violations"])


@router.get("/")
def get_violations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Get all violations for the current user with worker and camera info.
    """
    violations = (
        db.query(
            Violation,
            Worker.fullName,
            Camera.name.label("camera_name"),
            Camera.location.label("camera_location")
        )
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
            "created_at": v.Violation.created_at,
            "status": v.Violation.status,
            "snapshot": v.Violation.snapshot.decode("utf-8") if v.Violation.snapshot else None,
        }
        for v in violations
    ]


@router.post("/")
def create_violation(
    violation: ViolationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new violation and broadcast a notification to connected WebSocket clients.
    """
    # Create violation
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

    # Create notification in DB
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

    # Prepare notification data for WebSocket
    worker = db.query(Worker).filter(Worker.id == new_violation.worker_id).first()
    camera = db.query(Camera).filter(Camera.id == new_violation.camera_id).first()

    notification_data = {
        "id": new_notification.id,
        "violation_id": new_violation.id, 
        "message": message,
        "is_read": new_notification.is_read,
        "created_at": str(new_notification.created_at),
        "violation_type": new_violation.violation_types,
        "worker_code": new_violation.worker_code,
        "worker_name": worker.fullName if worker else "Unknown Worker",
        "camera": camera.name if camera else "Unknown Camera",
        "camera_location": camera.location if camera else "Unknown Location",
         "status": new_violation.status,   # include status for real-time updates,
    }

    # Broadcast via WebSocket
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast_notification(current_user.id, notification_data))
    except RuntimeError:
        asyncio.run(broadcast_notification(current_user.id, notification_data))

    return new_violation


@router.put("/{violation_id}/status")
async def update_violation_status(
    violation_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the status of a violation and broadcast the status change via WebSocket.
    """
    violation = (
        db.query(Violation)
        .filter(Violation.id == violation_id, Violation.user_id == current_user.id)
        .first()
    )
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")

    new_status = payload.get("status")
    if new_status not in ["pending", "resolved", "false positive"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    violation.status = new_status
    db.commit()
    db.refresh(violation)

    # Prepare WebSocket notification
    notification_data = {
        "violation_id": violation.id,
        "status": violation.status,
        "worker_code": violation.worker_code,
        "worker_name": violation.worker.fullName if violation.worker else "Unknown Worker",
        "message": f"Violation status updated to {violation.status}",
    }

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast_notification(current_user.id, notification_data))
    except RuntimeError:
        asyncio.run(broadcast_notification(current_user.id, notification_data))

    return {"message": "Status updated", "status": violation.status}

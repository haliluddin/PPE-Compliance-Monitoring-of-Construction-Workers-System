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
from datetime import datetime, timezone
import base64

router = APIRouter(prefix="/violations", tags=["Violations"])


def _format_camera_display(camera_name, camera_location):
    if not camera_name and not camera_location:
        return "Video Upload (Video Upload)"
    if camera_name and camera_location:
        return f"{camera_name} ({camera_location})"
    if camera_name:
        return f"{camera_name} ({camera_name})" if camera_name.lower().startswith("video upload") else camera_name
    return f"{camera_location} ({camera_location})"


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
        .outerjoin(Camera, Violation.camera_id == Camera.id)
        .filter(Violation.user_id == current_user.id)
        .all()
    )

    out = []
    for v in violations:
        snap_b64 = None
        if v.Violation.snapshot:
            try:
                snap_b64 = base64.b64encode(v.Violation.snapshot).decode("ascii")
            except Exception:
                snap_b64 = None

        camera_display = _format_camera_display(v.camera_name, v.camera_location)
        created_at_iso = None
        if getattr(v.Violation, "created_at", None):
            created_at_iso = v.Violation.created_at.isoformat()

        out.append({
            "id": v.Violation.id,
            "violation": v.Violation.violation_types,
            "worker": v.fullName,
            "worker_code": v.Violation.worker_code,
            "camera": camera_display,
            "created_at": created_at_iso,
            "status": v.Violation.status,
            "snapshot": snap_b64,
        })
    return out


@router.post("/")
def create_violation(
    violation: ViolationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new violation and broadcast a notification to connected WebSocket clients.
    """
    # Parse frame_ts safely into datetime (if provided)
    frame_ts_val = None
    if violation.frame_ts:
        try:
            # accept ISO format or unix timestamp string/int
            try:
                frame_ts_val = datetime.fromisoformat(str(violation.frame_ts))
            except Exception:
                try:
                    # try as unix seconds
                    ts = float(violation.frame_ts)
                    frame_ts_val = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    frame_ts_val = None
        except Exception:
            frame_ts_val = None

    new_violation = Violation(
        violation_types=violation.violation_types,
        worker_id=violation.worker_id,
        frame_ts=frame_ts_val,
        worker_code=violation.worker_code,
        camera_id=violation.camera_id,
        user_id=current_user.id,
        status="pending",
    )
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)

    # Create notification in DB (set created_at explicitly so it's full datetime)
    message = f"New violation detected: {new_violation.violation_types} by worker code {new_violation.worker_code}"
    new_notification = Notification(
        message=message,
        user_id=current_user.id,
        violation_id=new_violation.id,
        is_read=False,
        created_at=datetime.utcnow()
    )
    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)

    # Prepare notification data for WebSocket
    worker = db.query(Worker).filter(Worker.id == new_violation.worker_id).first()
    camera = db.query(Camera).filter(Camera.id == new_violation.camera_id).first()

    notification_data = {
         "type": "new_violation",
        "id": new_notification.id,
        "violation_id": new_violation.id,
        "message": message,
        "is_read": new_notification.is_read,
        "created_at": new_notification.created_at.isoformat(),
        "violation_type": new_violation.violation_types,
        "worker_code": new_violation.worker_code,
        "worker_name": worker.fullName if worker else "Unknown Worker",
        "camera": camera.name if camera else None,
        "camera_location": camera.location if camera else None,
         "status": new_violation.status,
    }

    # Broadcast via WebSocket (unchanged)
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
        "type": "status_update",
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

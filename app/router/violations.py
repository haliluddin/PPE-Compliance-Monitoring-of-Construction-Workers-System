from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Violation, Worker, User, Camera, Notification
from app.router.auth import get_current_user
from app.router.notifications_ws import broadcast_notification
from app.schemas import ViolationCreate
import asyncio
from datetime import datetime, timezone, timedelta
import base64

PH_TZ = timezone(timedelta(hours=8))

router = APIRouter(prefix="/violations", tags=["Violations"])

def _format_camera_display(camera_name, camera_location):
    if not camera_name and not camera_location:
        return "Video Upload (Video Upload)"
    if camera_name and camera_location:
        return f"{camera_name} ({camera_location})"
    if camera_name:
        return camera_name
    return camera_location

@router.get("/")
def get_violations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
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
            try:
                if v.Violation.created_at.tzinfo is None:
                    created_at_iso = v.Violation.created_at.replace(tzinfo=timezone.utc).astimezone(PH_TZ).isoformat()
                else:
                    created_at_iso = v.Violation.created_at.astimezone(PH_TZ).isoformat()
            except Exception:
                created_at_iso = v.Violation.created_at.isoformat()
        out.append({
            "id": v.Violation.id,
            "violation_types": v.Violation.violation_types,
            "worker": v.fullName,
            "worker_code": v.Violation.worker_code,
            "camera": camera_display,
            "camera_name": v.camera_name,
            "camera_location": v.camera_location,
            "created_at": created_at_iso,
            "status": v.Violation.status,
            "snapshot": snap_b64,
        })
    return out

@router.post("/")
def create_violation(violation: ViolationCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Parse frame timestamp (store as UTC if possible)
    frame_ts_val = None
    if violation.frame_ts:
        try:
            try:
                frame_ts_val = datetime.fromisoformat(str(violation.frame_ts))
            except Exception:
                try:
                    ts = float(violation.frame_ts)
                    frame_ts_val = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    frame_ts_val = None
        except Exception:
            frame_ts_val = None
    if frame_ts_val:
        if frame_ts_val.tzinfo is None:
            # assume it's UTC if no tzinfo
            frame_ts_val = frame_ts_val.replace(tzinfo=timezone.utc)
        # normalize to UTC
        frame_ts_val = frame_ts_val.astimezone(timezone.utc)

    # Use UTC for created_at (store canonical server time)
    now_utc = datetime.now(timezone.utc)

    new_violation = Violation(
        violation_types=violation.violation_types,
        worker_id=violation.worker_id,
        frame_ts=frame_ts_val,
        worker_code=violation.worker_code,
        camera_id=violation.camera_id,
        user_id=current_user.id,
        status="pending",
        created_at=now_utc,
    )
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)

    message = f"New violation: {new_violation.violation_types} by worker code {new_violation.worker_code}"
    new_notification = Notification(
        message=message,
        user_id=current_user.id,
        violation_id=new_violation.id,
        is_read=False,
        created_at=now_utc
    )
    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)

    worker = db.query(Worker).filter(Worker.id == new_violation.worker_id).first()
    cam = db.query(Camera).filter(Camera.id == new_violation.camera_id).first()
    if cam:
        camera_display = f"{cam.name} ({cam.location})" if cam.name and cam.location else (cam.name or cam.location)
        camera_name = cam.name
        camera_location = cam.location
    else:
        camera_display = "Video Upload (Video Upload)"
        camera_name = None
        camera_location = None

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
        "camera": camera_name,
        "camera_location": camera_location,
        "camera_display": camera_display,
        "status": new_violation.status,
    }
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast_notification(current_user.id, notification_data))
    except RuntimeError:
        asyncio.run(broadcast_notification(current_user.id, notification_data))
    return new_violation

@router.put("/{violation_id}/status")
async def update_violation_status(violation_id: int, payload: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    if violation.user_id != current_user.id and not getattr(current_user, "is_supervisor", False):
        raise HTTPException(status_code=403, detail="Not authorized to update this violation")
    new_status = payload.get("status")
    if new_status not in ["pending", "resolved", "false positive"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    # Update status and resolved_at accordingly (store resolved_at in UTC)
    violation.status = new_status
    now_utc = datetime.now(timezone.utc)
    if new_status.lower() == "resolved":
        violation.resolved_at = now_utc
    else:
        # clear resolved_at if status changed back to pending/false positive
        violation.resolved_at = None

    db.commit()
    db.refresh(violation)

    # create notification for status update
    status_message = f"Violation #{violation.id} status updated to {new_status}"
    status_notification = Notification(
        message=status_message,
        user_id=violation.user_id or current_user.id,
        violation_id=violation.id,
        is_read=False,
        created_at=now_utc
    )
    db.add(status_notification)
    db.commit()
    db.refresh(status_notification)

    cam = db.query(Camera).filter(Camera.id == violation.camera_id).first()
    if cam:
        camera_display = f"{cam.name} ({cam.location})" if cam.name and cam.location else (cam.name or cam.location)
        camera_name = cam.name
        camera_location = cam.location
    else:
        camera_display = "Video Upload (Video Upload)"
        camera_name = None
        camera_location = None

    notification_data = {
        "type": "status_update",
        "violation_id": violation.id,
        "status": violation.status,
        "message": status_message,
        "created_at": status_notification.created_at.isoformat(),
        "camera_display": camera_display,
        "camera": camera_name,
        "camera_location": camera_location,
        "violation_types": violation.violation_types,
        "worker_code": violation.worker_code,
    }
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast_notification(violation.user_id or current_user.id, notification_data))
    except RuntimeError:
        asyncio.run(broadcast_notification(violation.user_id or current_user.id, notification_data))
    return {"message": "Status updated", "status": violation.status, "violation_id": violation.id, "created_at": status_notification.created_at.isoformat()}

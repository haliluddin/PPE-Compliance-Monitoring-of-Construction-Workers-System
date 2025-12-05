from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Notification, Violation, Worker, Camera, User
from app.router.auth import get_current_user
from app.router.notifications_ws import broadcast_notification
import base64
from datetime import datetime, timezone, timedelta

PH_TZ = timezone(timedelta(hours=8))

def _format_camera_display(camera_name, camera_location):
    if not camera_name and not camera_location:
        return "Video Upload (Video Upload)"
    if camera_name and camera_location:
        return f"{camera_name} ({camera_location})"
    if camera_name:
        return camera_name
    return camera_location

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("/")
def get_notifications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    notifications = (
        db.query(
            Notification,
            Violation,
            Worker,
            Camera.name.label("camera_name"),
            Camera.location.label("camera_location"),
        )
        .outerjoin(Violation, Notification.violation_id == Violation.id)
        .outerjoin(Worker, Violation.worker_id == Worker.id)
        .outerjoin(Camera, Violation.camera_id == Camera.id)
        .filter(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(50)
        .all()
    )
    out = []
    for n in notifications:
        snap_b64 = None
        vobj = n.Violation
        if vobj and getattr(vobj, "snapshot", None):
            try:
                snap_b64 = base64.b64encode(vobj.snapshot).decode("ascii")
            except Exception:
                snap_b64 = None
        camera_display = _format_camera_display(getattr(n, "camera_name", None), getattr(n, "camera_location", None))
        created_iso = n.Notification.created_at.isoformat() if n.Notification.created_at else None
        out.append({
            "id": n.Notification.id,
            "violation_id": vobj.id if vobj else None,
            "message": n.Notification.message,
            "is_read": n.Notification.is_read,
            "created_at": created_iso,
            "camera_display": camera_display,
            "worker_code": getattr(vobj, "worker_code", None) if vobj else None,
            "worker_name": getattr(n.Worker, "fullName", "Unknown Worker"),
            "violation_type": getattr(vobj, "violation_types", "Unknown Violation"),
            "camera": camera_display,
            "camera_name": getattr(n, "camera_name", None),
            "camera_location": getattr(n, "camera_location", None),
            "status": getattr(vobj, "status", "Pending"),
            "snapshot": snap_b64,
        })
    return out

@router.post("/")
def create_notification_example_example():  # kept as placeholder route not used; actual notifications created by other routers
    raise HTTPException(status_code=404, detail="Not used")

@router.post("/mark_read/{notification_id}")
def mark_notification_read(notification_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    notification.is_read = True
    db.commit()
    db.refresh(notification)
    return {"success": True, "notification_id": notification_id}

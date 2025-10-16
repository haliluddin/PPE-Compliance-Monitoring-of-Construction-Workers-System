# app/router/notifications.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Notification, Violation, Worker, Camera, User
from app.router.auth import get_current_user

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("/")
def get_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # use outerjoin for safe left joins
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
        .all()
    )

    return [
        {
            "id": n.Notification.id,
            "message": n.Notification.message,
            "is_read": n.Notification.is_read,
            "created_at": n.Notification.created_at,
            "worker_code": getattr(n.Violation, "worker_code", None),
            "worker_name": getattr(n.Worker, "fullName", "Unknown Worker"),
            "violation_type": getattr(n.Violation, "violation_types", "Unknown Violation"),
            "camera": getattr(n, "camera_name", None) or "Unknown Camera",
            "camera_location": getattr(n, "camera_location", None) or "Unknown Location",
        }
        for n in notifications
    ]
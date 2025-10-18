# app/router/notifications.py
from fastapi import APIRouter, Depends, HTTPException
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
        .limit(10) 
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
             "status": getattr(n.Violation, "status", "Pending"),
        }
        for n in notifications
    ]


@router.post("/{notification_id}/mark_read")
def mark_notification_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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

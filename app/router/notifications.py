# app/router/notifications.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Notification, Violation, Worker, User
from app.router.auth import get_current_user

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("/")
def get_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    notifications = (
        db.query(Notification, Violation, Worker)
        .join(Violation, Notification.violation_id == Violation.id)
        .join(Worker, Violation.worker_id == Worker.id)
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
            "worker_code": n.Violation.worker_code,
            "worker_name": n.Worker.fullName,
            "violation_type": n.Violation.violation_types,
        }
        for n in notifications
    ]

# # app/router/notifications.py

# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from app.database import get_db
# from app.models import Notification, User
# from app.router.auth import get_current_user

# router = APIRouter(prefix="/notifications", tags=["Notifications"])

# @router.get("/")
# def get_notifications(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     notifications = (
#         db.query(Notification)
#         .filter(Notification.user_id == current_user.id)
#         .order_by(Notification.date_created.desc())
#         .all()
#     )
#     return [
#         {
#             "id": n.id,
#             "title": n.title,
#             "message": n.message,
#             "is_read": n.is_read,
#             "date_created": n.date_created,
#         }
#         for n in notifications
#     ]

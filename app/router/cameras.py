# app/routers/cameras.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Camera, User
from app.router.auth import get_current_user

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("/")
def get_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    cameras = db.query(Camera).filter(Camera.user_id == current_user.id).all()
    return [{"id": c.id, "name": c.name, "location": c.location} for c in cameras]

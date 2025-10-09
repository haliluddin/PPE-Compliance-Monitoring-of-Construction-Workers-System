# app/router/cameras.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Camera  # your SQLAlchemy Camera model

router = APIRouter(
    prefix="/cameras",
    tags=["cameras"],
)

# Schema for creating a camera
class CameraCreateSchema(BaseModel):
    name: str
    ip_address: str
    rtsp_url: str
    username: str = ""
    password: str = ""

# GET all cameras
@router.get("/")
async def get_cameras(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Camera))
    cameras = result.scalars().all()
    return cameras

# POST a new camera
@router.post("/")
async def add_camera(data: CameraCreateSchema, db: AsyncSession = Depends(get_db)):
    new_camera = Camera(
        name=data.name,
        ip_address=data.ip_address,
        rtsp_url=data.rtsp_url,
        username=data.username,
        password=data.password,
        status="LIVE"
    )
    db.add(new_camera)
    await db.commit()
    await db.refresh(new_camera)
    return new_camera

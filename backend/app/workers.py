# app/routers/workers.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Worker, User
from app.auth import get_current_user

router = APIRouter()

# Get workers for current user
@router.get("/workers")
async def get_workers(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Worker).where(Worker.user_id == current_user.id))
    return result.scalars().all()

# Add worker
@router.post("/workers")
async def add_worker(worker: WorkerCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    new_worker = Worker(**worker.dict(), user_id=current_user.id)
    db.add(new_worker)
    await db.commit()
    await db.refresh(new_worker)
    return new_worker

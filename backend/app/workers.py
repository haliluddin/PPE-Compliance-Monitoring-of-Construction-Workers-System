from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Worker
from app.schemas import WorkerCreate, WorkerResponse
from app.auth import get_current_user
from typing import List

router = APIRouter()

# GET workers for current user
@router.get("/workers", response_model=List[WorkerResponse])
async def get_workers(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Worker).where(Worker.user_id == current_user.id))
    return result.scalars().all()

# POST add worker
@router.post("/workers", response_model=WorkerResponse)
async def add_worker(worker: WorkerCreate, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    new_worker = Worker(**worker.dict(), user_id=current_user.id)
    db.add(new_worker)
    await db.commit()
    await db.refresh(new_worker)
    return new_worker

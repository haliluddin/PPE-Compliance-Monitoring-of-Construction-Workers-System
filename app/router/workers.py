from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models import Worker, Violation
from app.schemas import WorkerResponse
from app.router.auth import get_current_user
from typing import List

router = APIRouter()

# -----------------------------
# GET workers for current user (with total incidents)
# -----------------------------
@router.get("/workers")
def get_workers(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get workers with their total number of violations
    workers = (
        db.query(
            Worker.id,
            Worker.fullName,
            Worker.worker_code,
            Worker.assignedLocation,
            Worker.role,
            Worker.dateAdded,
            Worker.status,
            Worker.registered,
            Worker.user_id,
            func.count(Violation.id).label("totalIncidents")
        )
        .outerjoin(Violation, Violation.worker_id == Worker.id)
        .filter(Worker.user_id == current_user.id)
        .group_by(Worker.id)
        .all()
    )

    # Convert to list of dicts (for frontend)
    return [
        {
            "id": w.id,
            "fullName": w.fullName,
            "worker_code": w.worker_code,
            "assignedLocation": w.assignedLocation,
            "role": w.role,
            "dateAdded": w.dateAdded,
            "status": w.status,
            "registered": w.registered,
            "user_id": w.user_id,
            "totalIncidents": w.totalIncidents,
        }
        for w in workers
    ]

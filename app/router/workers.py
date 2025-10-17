# app/router/workers.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models import Worker, Violation, Camera
from app.schemas import WorkerResponse, WorkerCreate
from app.router.auth import get_current_user
from typing import List
from fastapi import HTTPException

router = APIRouter()



@router.get("/workers")
def get_workers(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
 
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
            func.count(Violation.id).label("totalIncidents"),
            func.max(Violation.frame_ts).label("lastSeen")
        )
        .outerjoin(Violation, Violation.worker_id == Worker.id)
        .filter(Worker.user_id == current_user.id)
        .group_by(Worker.id)
        .all()
    )

   
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
             "lastSeen": w.lastSeen.isoformat() if w.lastSeen else None
        }
        for w in workers
    ]


# -----------------------------
# POST add worker
# -----------------------------
@router.post("/workers", response_model=WorkerResponse)
def add_worker(
    worker: WorkerCreate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_worker = Worker(**worker.dict(), user_id=current_user.id)
    db.add(new_worker)
    db.commit()
    db.refresh(new_worker)
    return new_worker


# -----------------------------
# GET single worker with violations
# -----------------------------
@router.get("/workers/{worker_id}")
def get_worker_profile(
    worker_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get worker for current user
    worker = db.query(Worker).filter(
        Worker.id == worker_id,
        Worker.user_id == current_user.id
    ).first()
    
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

 
    violations = (
        db.query(
            Violation.id,
            Violation.frame_ts,
            Violation.violation_types,
            Camera.name.label("camera_name"),
            Camera.location.label("camera_location")
        )
        .join(Camera, Violation.camera_id == Camera.id, isouter=True)
        .filter(Violation.worker_id == worker.id)
        .all()
    )

    
    violation_history = [
        {
            "id": v.id,
            "date": v.frame_ts, 
            "type": v.violation_types,
            "cameraLocation": f"{v.camera_name or 'Unknown'} - {v.camera_location or 'N/A'}" 
        }
        for v in violations
    ]

    return {
        "id": worker.id,
        "fullName": worker.fullName,
        "worker_code": worker.worker_code,
        "assignedLocation": worker.assignedLocation,
        "role": worker.role,
        "dateAdded": worker.dateAdded,
        "status": worker.status,
        "registered": worker.registered,
        "user_id": worker.user_id,
        "totalIncidents": len(violation_history),
        "violationHistory": violation_history
    }

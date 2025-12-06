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
import base64
from datetime import datetime

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
            # Worker.assignedLocation,
            # Worker.role,
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
            "dateAdded": w.dateAdded,
            "status": w.status,
            "registered": w.registered,
            "user_id": w.user_id,
            "totalIncidents": w.totalIncidents,
             "lastSeen": (w.lastSeen.replace(tzinfo=timezone.utc).astimezone(PH_TZ).isoformat() if w.lastSeen and w.lastSeen.tzinfo is None else (w.lastSeen.astimezone(PH_TZ).isoformat() if w.lastSeen else None))
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
    # Check if worker_code already exists for the current user
    existing_worker = db.query(Worker).filter(
        Worker.worker_code == worker.worker_code,
        Worker.user_id == current_user.id
    ).first()

    if existing_worker:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Worker code '{worker.worker_code}' already exists."
        )

    # Create new worker
    new_worker = Worker(**worker.dict(), user_id=current_user.id)
    db.add(new_worker)
    db.commit()
    db.refresh(new_worker)
    return new_worker


# -----------------------------
# GET single worker with violations
# -----------------------------
def _format_camera_display(camera_name, camera_location):
    if not camera_name and not camera_location:
        return "Video Upload (Video Upload)"
    if camera_name and camera_location:
        return f"{camera_name} ({camera_location})"
    if camera_name:
        return camera_name
    return camera_location

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
            Violation.created_at,
            Violation.violation_types,
            Violation.status,
            Camera.name.label("camera_name"),
            Camera.location.label("camera_location"),
            Violation.snapshot
        )
        .outerjoin(Camera, Violation.camera_id == Camera.id)
        .filter(Violation.worker_id == worker.id)
        .all()
    )

    violation_history = []
    for v in violations:
        snap_b64 = None
        if v.snapshot:
            try:
                snap_b64 = base64.b64encode(v.snapshot).decode("ascii")
            except Exception:
                snap_b64 = None
        camera_display = _format_camera_display(v.camera_name, v.camera_location)
        created_iso = v.created_at.isoformat() if v.created_at else None
        violation_history.append({
            "id": v.id,
            "date": created_iso,
            "type": v.violation_types,
            "status": v.status,
            "cameraLocation": camera_display,
            "worker_name": worker.fullName,
            "snapshot": snap_b64
        })

    return {
        "id": worker.id,
        "fullName": worker.fullName,
        "worker_code": worker.worker_code,
        "dateAdded": worker.dateAdded,
        "status": worker.status,
        "registered": worker.registered,
        "user_id": worker.user_id,
        "totalIncidents": len(violation_history),
        "violationHistory": violation_history
    }
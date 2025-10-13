# app/routers/violations.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Violation, Worker, User
from app.router.auth import get_current_user
from app.schemas import ViolationCreate, ViolationResponse

router = APIRouter(prefix="/violations", tags=["Violations"])

@router.get("/")
def get_violations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    violations = (
        db.query(Violation, Worker.fullName)
        .join(Worker, Violation.worker_id == Worker.id)
        .filter(Violation.user_id == current_user.id)  
        .all()
    )

    return [
        {
            "id": v.Violation.id,
            "violation": v.Violation.violation_types,
            "worker": v.fullName,
            "worker_code": v.Violation.worker_code,
            "frame_ts": v.Violation.frame_ts,
            "status": v.Violation.status,
        }
        for v in violations
    ]


@router.post("/")
def create_violation(
    violation: ViolationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_violation = Violation(
        violation_types=violation.violation_types,
        worker_id=violation.worker_id,
        frame_ts=violation.frame_ts,
        worker_code=violation.worker_code,
        user_id=current_user.id,  
        status="pending",
    )
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)
    return new_violation


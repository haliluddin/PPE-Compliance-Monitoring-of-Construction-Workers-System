# app/routers/violations.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Violation, Worker

router = APIRouter(prefix="/violations", tags=["Violations"])

@router.get("/")
def get_violations(db: Session = Depends(get_db)):
    violations = (
        db.query(Violation, Worker.fullName)
        .join(Worker, Violation.worker_id == Worker.id)
        .all()
    )
    return [
        {
            "id": v.Violation.id,
            "violation": v.Violation.violation_types,
            "worker": v.fullName,
            "worker_code": v.Violation.worker_code,
            "frame_ts": v.Violation.frame_ts,
        }
        for v in violations
    ]

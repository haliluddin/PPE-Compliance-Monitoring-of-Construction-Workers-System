from sqlalchemy import Column, Integer, Text, DateTime, JSON, LargeBinary, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    location = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True)
    worker_code = Column(Text, unique=True)
    registered = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    job_type = Column(Text, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    status = Column(Text, default="queued")
    meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    worker_id = Column(Integer, ForeignKey("workers.id"))
    worker_code = Column(Text)
    violation_types = Column(Text)
    frame_index = Column(Integer)
    frame_ts = Column(DateTime(timezone=True))
    snapshot = Column(LargeBinary)
    inference = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

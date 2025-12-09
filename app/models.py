# app/models.py
from sqlalchemy import Column, Integer, String, Date, Text, DateTime, JSON, LargeBinary, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_supervisor = Column(Boolean, default=False)
    workers = relationship("Worker", back_populates="user")
    violations = relationship("Violation", back_populates="user")
    cameras = relationship("Camera", back_populates="user")
    jobs = relationship("Job", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    location = Column(Text)
    stream_url = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="cameras")
    jobs = relationship("Job", back_populates="camera")
    violations = relationship("Violation", back_populates="camera")

class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True, index=True)
    fullName = Column(String, nullable=False)
    worker_code = Column(String, nullable=False)
    dateAdded = Column(Date, nullable=True, server_default=func.current_date())
    status = Column(String, nullable=False, server_default="Active")
    registered = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="workers")
    violations = relationship("Violation", back_populates="worker")

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    job_type = Column(Text, nullable=False)
    status = Column(Text, default="queued")
    meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="jobs")
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    camera = relationship("Camera", back_populates="jobs")
    violations = relationship("Violation", back_populates="job")

class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=True)
    worker_code = Column(Text)
    violation_types = Column(Text)
    frame_index = Column(Integer)
    frame_ts = Column(DateTime(timezone=True))
    snapshot = Column(LargeBinary)
    inference = Column(JSON)
    status = Column(String, nullable=False, server_default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)   # <-- NEW column
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    worker = relationship("Worker", back_populates="violations")
    user = relationship("User", back_populates="violations")
    camera = relationship("Camera", back_populates="violations")
    job = relationship("Job", back_populates="violations")
    notification = relationship("Notification", back_populates="violation", uselist=False)

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="notifications")
    violation_id = Column(Integer, ForeignKey("violations.id"), nullable=True)
    violation = relationship("Violation", back_populates="notification")

# app/models.py
from sqlalchemy import Column, Integer, Text, DateTime, JSON, LargeBinary, Boolean, ForeignKey, Column, Integer, String, Date, Text, TIMESTAMP
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from .database import Base

# Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_supervisor = Column(Boolean, default=False)
    workers = relationship("Worker", back_populates="user")

# class Camera(Base):
#     __tablename__ = "cameras"
#     id = Column(Integer, primary_key=True)
#     name = Column(Text, nullable=False)
#     location = Column(Text)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())

class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True, index=True)
    fullName = Column(String, nullable=False)
    worker_code = Column(String, nullable=False)  
    assignedLocation = Column(String, nullable=False)
    role = Column(String, nullable=False)
    dateAdded = Column(Date, nullable=False)
    status = Column(String, nullable=False)
    registered = Column(Boolean, default=False)  

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="workers")
    
# class Job(Base):
#     __tablename__ = "jobs"
#     id = Column(Integer, primary_key=True)
#     job_type = Column(Text, nullable=False)
#     camera_id = Column(Integer, ForeignKey("cameras.id"))
#     status = Column(Text, default="queued")
#     meta = Column(JSON)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     started_at = Column(DateTime(timezone=True), nullable=True)
#     finished_at = Column(DateTime(timezone=True), nullable=True)

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

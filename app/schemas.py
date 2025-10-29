# app/schemas.py
from pydantic import BaseModel
from datetime import date

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserOut(BaseModel):
    id: int
    username: str
    role: str

    class Config:
        from_attributes = True


class WorkerCreate(BaseModel):
    fullName: str
    worker_code: str
    # assignedLocation: str
    # role: str
    dateAdded: date
    status: str
    registered: bool = True  

class WorkerResponse(BaseModel):
    id: int
    fullName: str
    worker_code: str
    # assignedLocation: str
    # role: str
    dateAdded: date
    status: str
    registered: bool 
    user_id: int
    totalIncidents: int = 0 

    class Config:
        orm_mode = True

class ViolationCreate(BaseModel):
    violation_types: str
    worker_id: int
    worker_code: str
    camera_id: int 
    frame_ts: str 

class ViolationResponse(BaseModel):
    id: int
    violation_types: str
    worker_id: int
    worker_code: str
    frame_ts: str
    statud:str
    user_id: int

    class Config:
        orm_mode = True

class StatusUpdate(BaseModel):
    status: str
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
    workerNumber: str
    assignedLocation: str
    role: str
    dateAdded: date
    status: str

class WorkerOut(WorkerCreate):
    id: int
    user_id: int

    class Config:
        orm_mode = True


class WorkerResponse(WorkerCreate):
    id: int
    class Config:
        orm_mode = True
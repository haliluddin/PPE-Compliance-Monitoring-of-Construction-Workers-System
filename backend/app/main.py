from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "postgresql+asyncpg://postgres:ppe@localhost:5432/ppe_compliance"
engine = create_async_engine(DATABASE_URL, echo=False, future=True)


@app.get("/")
async def read_root():
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT current_database();"))
            db_name = result.scalar()
        return {
            "message": "PPE Compliance Monitoring System is running ",
            "database": db_name
        }
    except Exception as e:
        return {
            "message": "Backend running but DB connection failed ",
            "error": str(e)
        }

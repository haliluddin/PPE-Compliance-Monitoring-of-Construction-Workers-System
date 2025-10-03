from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+asyncpg://postgres:ppe@localhost:5432/ppe_compliance"

# Async engine
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Async session factory
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Dependency for FastAPI
async def get_db():
    async with SessionLocal() as session:
        yield session

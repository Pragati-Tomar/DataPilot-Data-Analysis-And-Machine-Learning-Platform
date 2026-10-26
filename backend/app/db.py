"""Optional PostgreSQL metadata and operation-history persistence."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = None
SessionLocal = None
DB_AVAILABLE = False
if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        DB_AVAILABLE = True
    except Exception:
        pass

AUTH_STORE_DIR = Path(__file__).resolve().parents[1] / "data_store"
AUTH_STORE_DIR.mkdir(parents=True, exist_ok=True)
if DB_AVAILABLE and engine is not None:
    AuthEngine = engine
else:
    AuthEngine = create_engine(
        f"sqlite:///{AUTH_STORE_DIR / 'auth.sqlite3'}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
AuthSessionLocal = sessionmaker(bind=AuthEngine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[int] = mapped_column(Integer)
    owner_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class OperationRecord(Base):
    __tablename__ = "dataset_operations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    operation: Mapped[str] = mapped_column(String(100))
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class UserRecord(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

def init_db() -> None:
    Base.metadata.create_all(bind=AuthEngine)
    if DB_AVAILABLE and engine is not None:
        # create_all does not alter existing tables, so upgrade existing installs.
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS owner_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_datasets_owner_id ON datasets (owner_id)"))

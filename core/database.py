"""
core/database.py
================
SQLAlchemy ORM layer for Vision-Track.

Supports two backends controlled via the DATABASE_URL environment variable:

  SQLite (default, zero-config):
      DATABASE_URL=sqlite:///outputs/violations.db   ← default value

  PostgreSQL (production / docker-compose):
      DATABASE_URL=postgresql://user:password@db:5432/visiontrack

Public API
----------
  init_db()       – Create all tables (idempotent, call once at startup).
  get_session()   – Context-manager that yields a transactional Session.
  ViolationRecord – ORM model for the 'violations' table.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Database URL — read from environment, fall back to a local SQLite file.
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "sqlite:///outputs/violations.db",
)


# ---------------------------------------------------------------------------
# ORM Base and models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class ViolationRecord(Base):
    """A single lane-violation event persisted to the database.

    Each row represents one (track_id, roi_zone) pair entering a restricted
    zone for the first time.  Duplicate-prevention is enforced at the
    application level in LaneViolationDetector, so every row here is unique.

    Columns
    -------
    id         – Auto-increment primary key.
    timestamp  – UTC datetime the violation was detected.
    track_id   – BoT-SORT numeric track identifier.
    class_name – COCO class label (e.g. 'car', 'truck').
    roi_zone   – 1-based restricted zone index.
    center_x   – Bounding-box centre X coordinate in pixels.
    center_y   – Bounding-box centre Y coordinate in pixels.
    """

    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    track_id = Column(Integer, nullable=False, index=True)
    class_name = Column(String(64), nullable=False)
    roi_zone = Column(Integer, nullable=False)  # 1-based for human readability
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ViolationRecord id={self.id} track_id={self.track_id} "
            f"zone={self.roi_zone} ts={self.timestamp}>"
        )


# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------


def _build_engine():
    """Construct a SQLAlchemy engine with backend-appropriate defaults.

    SQLite:
      - WAL journal mode (allows concurrent readers while a writer is active).
      - Foreign-key enforcement (off by default in SQLite).
      - NORMAL synchronous mode (safe but faster than FULL).

    PostgreSQL:
      - pool_pre_ping keeps stale connections from causing silent failures.
      - Relies on psycopg2-binary driver (included in requirements.txt).
    """
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        # Set DB_ECHO=1 in the environment to print every SQL statement.
        echo=os.environ.get("DB_ECHO", "0") == "1",
    )

    if DATABASE_URL.startswith("sqlite"):
        # SQLite pragmas must be applied on every new connection.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")  # concurrent read access
            cur.execute("PRAGMA foreign_keys=ON")  # enforce FK constraints
            cur.execute("PRAGMA synchronous=NORMAL")  # balance safety / speed
            cur.close()

    return engine


# Module-level singletons — instantiated once at import time.
_engine = _build_engine()
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all database tables if they do not already exist.

    This function is idempotent — safe to call multiple times (e.g. in
    both main.py and LaneViolationDetector).  For SQLite it also creates
    the parent directory of the database file.
    """
    if DATABASE_URL.startswith("sqlite:///"):
        # Strip 'sqlite:///' to obtain the filesystem path.
        db_file = Path(DATABASE_URL[len("sqlite:///") :])
        db_file.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional Session; commit on success, rollback on error.

    Usage::

        with get_session() as session:
            session.add(ViolationRecord(...))
        # automatically committed here

    The session is always closed in the ``finally`` block, so it is safe
    to use this inside a long-running loop.
    """
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

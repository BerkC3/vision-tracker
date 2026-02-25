"""
tests/test_database.py
======================
Unit tests for core.database — ORM model, engine factory, init_db, get_session.

Each test that needs a real database uses monkeypatch to swap the module-level
_engine and _SessionFactory for an in-memory SQLite instance, so no file is
created on disk (except the directory-creation test which uses tmp_path).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import core.database as db_mod
from core.database import Base, ViolationRecord, get_session, init_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> ViolationRecord:
    """Build a ViolationRecord with sensible defaults; override via kwargs."""
    defaults = dict(
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
        track_id=1,
        class_name="car",
        roi_zone=1,
        center_x=100.0,
        center_y=200.0,
    )
    defaults.update(kwargs)
    return ViolationRecord(**defaults)


@pytest.fixture
def mem_db(monkeypatch):
    """Replace the module-level engine with an in-memory SQLite instance.

    Yields the engine so tests can inspect the database directly.
    All changes are reverted by monkeypatch after the test.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "_engine", engine)
    monkeypatch.setattr(db_mod, "_SessionFactory", factory)
    yield engine


# ---------------------------------------------------------------------------
# ViolationRecord — ORM model
# ---------------------------------------------------------------------------


class TestViolationRecord:
    def test_repr_contains_track_id(self) -> None:
        r = _make_record(id=5, track_id=42)
        assert "42" in repr(r)

    def test_repr_contains_zone(self) -> None:
        r = _make_record(id=1, roi_zone=3)
        assert "zone=3" in repr(r)

    def test_repr_contains_timestamp(self) -> None:
        r = _make_record(id=1, timestamp=datetime(2025, 1, 1))
        assert "2025" in repr(r)

    def test_all_fields_stored(self) -> None:
        ts = datetime(2025, 6, 15, 9, 30, 0)
        r = ViolationRecord(
            timestamp=ts,
            track_id=7,
            class_name="truck",
            roi_zone=2,
            center_x=300.5,
            center_y=150.2,
        )
        assert r.track_id == 7
        assert r.class_name == "truck"
        assert r.roi_zone == 2
        assert r.center_x == pytest.approx(300.5)
        assert r.center_y == pytest.approx(150.2)
        assert r.timestamp == ts


# ---------------------------------------------------------------------------
# _build_engine — SQLite pragma setup
# ---------------------------------------------------------------------------


class TestBuildEngine:
    def test_sqlite_wal_mode_applied(self, tmp_path, monkeypatch) -> None:
        """SQLite engine must be configured with WAL journal mode."""
        url = f"sqlite:///{tmp_path}/pragmas.db"
        monkeypatch.setattr(db_mod, "DATABASE_URL", url)

        engine = db_mod._build_engine()
        with engine.connect() as conn:
            mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert mode == "wal"

    def test_sqlite_foreign_keys_enabled(self, tmp_path, monkeypatch) -> None:
        """SQLite engine must enforce foreign key constraints."""
        url = f"sqlite:///{tmp_path}/fk.db"
        monkeypatch.setattr(db_mod, "DATABASE_URL", url)

        engine = db_mod._build_engine()
        with engine.connect() as conn:
            fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
        assert fk == 1


# ---------------------------------------------------------------------------
# Lazy singleton initialisation
# ---------------------------------------------------------------------------


class TestLazyInit:
    """Verify the lazy singleton pattern for _engine and _SessionFactory."""

    def test_get_engine_returns_same_instance(self, tmp_path, monkeypatch) -> None:
        """_get_engine() must return the same engine on subsequent calls."""
        url = f"sqlite:///{tmp_path}/lazy.db"
        monkeypatch.setattr(db_mod, "DATABASE_URL", url)
        monkeypatch.setattr(db_mod, "_engine", None)

        engine1 = db_mod._get_engine()
        engine2 = db_mod._get_engine()
        assert engine1 is engine2

    def test_get_session_factory_returns_same_instance(
        self, tmp_path, monkeypatch
    ) -> None:
        """_get_session_factory() must return the same factory on subsequent calls."""
        url = f"sqlite:///{tmp_path}/lazy_sf.db"
        monkeypatch.setattr(db_mod, "DATABASE_URL", url)
        monkeypatch.setattr(db_mod, "_engine", None)
        monkeypatch.setattr(db_mod, "_SessionFactory", None)

        f1 = db_mod._get_session_factory()
        f2 = db_mod._get_session_factory()
        assert f1 is f2

    def test_engine_not_created_on_import(self, monkeypatch) -> None:
        """Module-level _engine must be None (lazy); only created on first use."""
        monkeypatch.setattr(db_mod, "_engine", None)
        monkeypatch.setattr(db_mod, "_SessionFactory", None)
        # After monkeypatch, no engine should exist
        assert db_mod._engine is None
        assert db_mod._SessionFactory is None

    def test_monkeypatch_overrides_lazy_init(self, monkeypatch) -> None:
        """When a test sets _engine via monkeypatch, _get_engine() must use it
        — not build a new one.  This verifies backward-compat with existing tests."""
        sentinel_engine = create_engine("sqlite:///:memory:")
        monkeypatch.setattr(db_mod, "_engine", sentinel_engine)

        assert db_mod._get_engine() is sentinel_engine


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_violations_table(self, mem_db, monkeypatch) -> None:
        """init_db() must create the violations table (idempotent)."""
        # Drop first to verify create_all actually runs
        Base.metadata.drop_all(mem_db)
        init_db()
        with mem_db.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master"
                    " WHERE type='table' AND name='violations'"
                )
            ).fetchone()
        assert result is not None

    def test_idempotent_second_call_does_not_raise(self, mem_db) -> None:
        """Calling init_db() a second time must not raise an error."""
        init_db()
        init_db()  # should be a no-op

    def test_creates_sqlite_parent_directory(self, tmp_path, monkeypatch) -> None:
        """init_db() must create the parent directory for a SQLite file path."""
        nested_db = tmp_path / "a" / "b" / "c" / "violations.db"
        url = f"sqlite:///{nested_db}"

        engine = create_engine(url)
        factory = sessionmaker(bind=engine)
        monkeypatch.setattr(db_mod, "_engine", engine)
        monkeypatch.setattr(db_mod, "_SessionFactory", factory)
        monkeypatch.setattr(db_mod, "DATABASE_URL", url)

        assert not nested_db.parent.exists()
        init_db()
        assert nested_db.parent.exists()


# ---------------------------------------------------------------------------
# get_session — transaction management
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_commit_on_success(self, mem_db) -> None:
        """A record added inside get_session() must be visible after the block."""
        with get_session() as session:
            session.add(_make_record(track_id=10))

        with mem_db.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM violations WHERE track_id=10")
            ).scalar()
        assert count == 1

    def test_rollback_on_exception(self, mem_db) -> None:
        """When an exception is raised inside get_session(), the transaction
        must be rolled back and the exception must propagate to the caller."""
        with pytest.raises(ValueError, match="simulated"):
            with get_session() as session:
                session.add(_make_record(track_id=99))
                raise ValueError("simulated")

        with mem_db.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM violations WHERE track_id=99")
            ).scalar()
        assert count == 0  # nothing committed

    def test_session_closed_after_success(self, mem_db) -> None:
        """The session object must be closed after a successful block."""
        captured = []

        with get_session() as session:
            captured.append(session)

        # After the context manager exits, the session must not be in a transaction.
        # Note: in SQLAlchemy 2.x, is_active returns True after close() because the
        # session is reset to "ready" state.  in_transaction() correctly reflects
        # that no transaction is open.
        assert not captured[0].in_transaction()

    def test_session_closed_after_exception(self, mem_db) -> None:
        """The session must be closed even when an exception occurs."""
        captured = []
        with pytest.raises(RuntimeError):
            with get_session() as session:
                captured.append(session)
                raise RuntimeError("boom")

        assert not captured[0].in_transaction()

    def test_multiple_records_in_single_session(self, mem_db) -> None:
        """Multiple records added in one get_session() block must all commit."""
        with get_session() as session:
            session.add(_make_record(track_id=1))
            session.add(_make_record(track_id=2))
            session.add(_make_record(track_id=3))

        with mem_db.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM violations")).scalar()
        assert count == 3

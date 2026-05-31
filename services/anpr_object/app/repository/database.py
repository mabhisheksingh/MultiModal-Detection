from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

import structlog

from triton_server.app import settings

logger = structlog.get_logger(__name__)

DATABASE_URL = settings.global_track_database_url

engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_schema() -> None:
    if not DATABASE_URL.startswith("sqlite"):
        return

    inspector = inspect(engine)
    if "global_identities" not in inspector.get_table_names():
        return

    column_names = {
        column["name"] for column in inspector.get_columns("global_identities")
    }
    if "license_plate_confidence" in column_names:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE global_identities "
                "ADD COLUMN license_plate_confidence FLOAT NOT NULL DEFAULT 0.0"
            )
        )
    logger.info(
        "global_tracking_db_schema_migrated",
        column="license_plate_confidence",
        table="global_identities",
    )


def init_db() -> None:
    """Create all ORM tables (idempotent)."""
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_schema()
    logger.info("global_tracking_db_initialized", url=DATABASE_URL)

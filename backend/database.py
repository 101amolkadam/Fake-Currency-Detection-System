"""Database configuration and session management."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

# Detect database type
IS_SQLITE = settings.DATABASE_URL.startswith("sqlite:///")

# Engine configuration based on database type
if IS_SQLITE:
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )
    
    # Enable foreign key constraints for MySQL
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set SQL mode for MySQL."""
        if not IS_SQLITE:
            cursor = dbapi_connection.cursor()
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()


def get_db():
    """FastAPI dependency for database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    import orm_models.analysis  # noqa: F401 - Import to register models
    Base.metadata.create_all(bind=engine)


def get_engine():
    """Get the SQLAlchemy engine instance."""
    return engine

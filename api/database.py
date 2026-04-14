"""SQLite persistence layer for underwriting records.

Uses SQLAlchemy Core (sync) so it works without asyncio complexity.
The DB file lives at data/underwriting.db next to the project root.

Tables
------
underwriting_records
    Every /underwrite call is stored here with inputs + full JSON response.
    Lets credit officers review history and lets us track model drift.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── DB location ──────────────────────────────────────────────────────────────
_DB_DIR = Path(__file__).resolve().parents[1] / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "underwriting.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ── ORM model ─────────────────────────────────────────────────────────────────
class UnderwritingRecord(Base):
    __tablename__ = "underwriting_records"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Inputs
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    shop_size_sqft = Column(Float, nullable=True)
    monthly_rent = Column(Float, nullable=True)
    years_in_operation = Column(Float, nullable=True)
    image_count = Column(Integer, nullable=False)
    image_names = Column(Text, nullable=True)   # JSON list of filenames

    # Core outputs
    recommendation = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    daily_sales_low = Column(Float, nullable=False)
    daily_sales_high = Column(Float, nullable=False)
    monthly_revenue_low = Column(Float, nullable=False)
    monthly_revenue_high = Column(Float, nullable=False)
    monthly_income_low = Column(Float, nullable=False)
    monthly_income_high = Column(Float, nullable=False)
    risk_flags = Column(Text, nullable=False)   # JSON array of flag codes
    shop_type = Column(String(50), nullable=True)
    tier = Column(String(20), nullable=True)

    # Full response blob (for audit / replay)
    response_json = Column(Text, nullable=False)

    # Performance
    processing_ms = Column(Integer, nullable=True)


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Write helper ──────────────────────────────────────────────────────────────
def save_record(
    db: Session,
    *,
    lat: float,
    lng: float,
    shop_size_sqft: float | None,
    monthly_rent: float | None,
    years_in_operation: float | None,
    image_names: list[str],
    response,          # UnderwriteResponse
    processing_ms: int,
) -> UnderwritingRecord:
    ex = response.explainability
    vf = ex.get("vision_features", {})
    gf = ex.get("geo_features", {})

    rec = UnderwritingRecord(
        lat=lat,
        lng=lng,
        shop_size_sqft=shop_size_sqft,
        monthly_rent=monthly_rent,
        years_in_operation=years_in_operation,
        image_count=len(image_names),
        image_names=json.dumps(image_names),
        recommendation=response.recommendation,
        confidence_score=response.confidence_score,
        daily_sales_low=response.daily_sales_range[0],
        daily_sales_high=response.daily_sales_range[1],
        monthly_revenue_low=response.monthly_revenue_range[0],
        monthly_revenue_high=response.monthly_revenue_range[1],
        monthly_income_low=response.monthly_income_range[0],
        monthly_income_high=response.monthly_income_range[1],
        risk_flags=json.dumps(response.risk_flags),
        shop_type=vf.get("shop_type"),
        tier=gf.get("tier"),
        response_json=response.model_dump_json(),
        processing_ms=processing_ms,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


# ── Read helpers ──────────────────────────────────────────────────────────────
def list_records(db: Session, limit: int = 50, offset: int = 0) -> list[UnderwritingRecord]:
    return (
        db.query(UnderwritingRecord)
        .order_by(UnderwritingRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_record(db: Session, record_id: int) -> UnderwritingRecord | None:
    return db.query(UnderwritingRecord).filter(UnderwritingRecord.id == record_id).first()


def get_stats(db: Session) -> dict:
    total = db.query(UnderwritingRecord).count()
    if total == 0:
        return {"total": 0, "approved": 0, "verify": 0, "rejected": 0}
    approved = db.query(UnderwritingRecord).filter(
        UnderwritingRecord.recommendation == "approve_for_underwriting"
    ).count()
    verify = db.query(UnderwritingRecord).filter(
        UnderwritingRecord.recommendation == "needs_verification"
    ).count()
    rejected = db.query(UnderwritingRecord).filter(
        UnderwritingRecord.recommendation == "reject_or_field_visit"
    ).count()
    return {"total": total, "approved": approved, "verify": verify, "rejected": rejected}

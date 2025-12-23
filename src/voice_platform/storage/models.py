"""Database models for voice platform."""
from datetime import date, time
from typing import Optional

from sqlalchemy import String, Date, Time, Text
from sqlalchemy.orm import Mapped, mapped_column

from db_utils import BaseModel, TimestampMixin


class Appointment(BaseModel, TimestampMixin):
    """Appointment model."""
    
    __tablename__ = "appointments"
    
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    caller_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    appointment_date: Mapped[date] = mapped_column(Date, nullable=False)
    appointment_time: Mapped[time] = mapped_column(Time, nullable=False)
    purpose: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="scheduled")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

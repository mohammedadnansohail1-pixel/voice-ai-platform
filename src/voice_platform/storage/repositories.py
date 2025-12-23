"""Repository implementations."""
from datetime import date
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db_utils import BaseRepository
from .models import Appointment


class AppointmentRepository(BaseRepository[Appointment]):
    """Repository for appointment operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Appointment)
    
    async def get_by_date(self, appointment_date: date) -> List[Appointment]:
        """Get all appointments for a specific date."""
        result = await self.session.execute(
            select(Appointment)
            .where(Appointment.appointment_date == appointment_date)
            .order_by(Appointment.appointment_time)
        )
        return list(result.scalars().all())
    
    async def get_by_session(self, session_id: str) -> List[Appointment]:
        """Get appointments by session ID."""
        result = await self.session.execute(
            select(Appointment)
            .where(Appointment.session_id == session_id)
            .order_by(Appointment.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_upcoming(self, limit: int = 10) -> List[Appointment]:
        """Get upcoming appointments."""
        from datetime import date as date_type
        result = await self.session.execute(
            select(Appointment)
            .where(Appointment.appointment_date >= date_type.today())
            .where(Appointment.status == "scheduled")
            .order_by(Appointment.appointment_date, Appointment.appointment_time)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def cancel(self, id: str) -> Optional[Appointment]:
        """Cancel an appointment."""
        return await self.update(id, status="cancelled")

"""Repository implementations."""
from datetime import date, time, datetime, timedelta
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
        result = await self.session.execute(
            select(Appointment)
            .where(Appointment.appointment_date >= date.today())
            .where(Appointment.status == "scheduled")
            .order_by(Appointment.appointment_date, Appointment.appointment_time)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def cancel(self, id: str) -> Optional[Appointment]:
        """Cancel an appointment."""
        return await self.update(id, status="cancelled")
    
    async def create_from_booking(
        self,
        session_id: str,
        day: str,
        time_str: str,
        reason: str,
        confirmation_number: str,
    ) -> Appointment:
        """
        Create appointment from agent booking.
        
        Args:
            session_id: Call session ID
            day: Day of week (e.g., "Tuesday")
            time_str: Time string (e.g., "2:00 PM")
            reason: Visit reason
            confirmation_number: Booking confirmation number
        """
        # Convert day of week to next occurrence date
        appointment_date = self._next_weekday(day)
        
        # Parse time string
        appointment_time = self._parse_time(time_str)
        
        appointment = await self.create(
            session_id=session_id,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            purpose=reason,
            status="scheduled",
            notes=f"Confirmation: {confirmation_number}. Booked via voice assistant.",
        )
        
        return appointment
    
    def _next_weekday(self, day_name: str) -> date:
        """Get the next occurrence of a weekday."""
        days = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        target = days.get(day_name.lower(), 0)
        today = date.today()
        days_ahead = target - today.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string like '2:00 PM' to time object."""
        from datetime import datetime
        
        # Try different formats
        for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M"]:
            try:
                return datetime.strptime(time_str.strip(), fmt).time()
            except ValueError:
                continue
        
        # Default to noon if parsing fails
        return time(12, 0)

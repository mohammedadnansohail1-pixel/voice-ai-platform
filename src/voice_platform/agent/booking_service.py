"""
Booking service - handles persistence of appointments.

Separates booking logic from agent logic for clean architecture.
"""
from typing import Optional, Dict, Any
from datetime import date, time, datetime, timedelta

from ..logging import get_logger

logger = get_logger("agent.booking_service")


class BookingService:
    """
    Service for persisting bookings to database.
    
    Can be used with or without database - gracefully degrades.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize booking service.
        
        Args:
            db_session: Optional async database session
        """
        self.db_session = db_session
        self._has_db = db_session is not None
        
    async def save_booking(
        self,
        session_id: str,
        day: str,
        time_str: str,
        reason: str,
        confirmation_number: str,
        caller_phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a booking to the database.
        
        Returns booking record or in-memory dict if no DB.
        """
        # Parse day to date
        appointment_date = self._next_weekday(day)
        appointment_time = self._parse_time(time_str)
        
        booking_data = {
            "session_id": session_id,
            "appointment_date": appointment_date.isoformat(),
            "appointment_time": appointment_time.strftime("%H:%M"),
            "day": day,
            "time": time_str,
            "purpose": reason,
            "status": "scheduled",
            "confirmation_number": confirmation_number,
            "caller_phone": caller_phone,
        }
        
        if self._has_db:
            try:
                from ..storage.repositories import AppointmentRepository
                repo = AppointmentRepository(self.db_session)
                
                appointment = await repo.create_from_booking(
                    session_id=session_id,
                    day=day,
                    time_str=time_str,
                    reason=reason,
                    confirmation_number=confirmation_number,
                )
                
                booking_data["id"] = str(appointment.id)
                booking_data["saved_to_db"] = True
                
                logger.info(
                    "booking_saved_to_db",
                    confirmation=confirmation_number,
                    id=str(appointment.id),
                )
                
            except Exception as e:
                logger.error("booking_db_error", error=str(e))
                booking_data["saved_to_db"] = False
                booking_data["db_error"] = str(e)
        else:
            booking_data["saved_to_db"] = False
            logger.info(
                "booking_in_memory_only",
                confirmation=confirmation_number,
            )
        
        return booking_data
    
    def _next_weekday(self, day_name: str) -> date:
        """Get the next occurrence of a weekday."""
        days = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        target = days.get(day_name.lower(), 0)
        today = date.today()
        days_ahead = target - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string like '2:00 PM' to time object."""
        for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M"]:
            try:
                return datetime.strptime(time_str.strip(), fmt).time()
            except ValueError:
                continue
        return time(12, 0)


# Sync wrapper for non-async contexts
def save_booking_sync(
    session_id: str,
    day: str,
    time_str: str,
    reason: str,
    confirmation_number: str,
) -> Dict[str, Any]:
    """
    Synchronous booking save (in-memory only).
    
    Use BookingService.save_booking() for database persistence.
    """
    service = BookingService(db_session=None)
    
    appointment_date = service._next_weekday(day)
    appointment_time = service._parse_time(time_str)
    
    return {
        "session_id": session_id,
        "appointment_date": appointment_date.isoformat(),
        "appointment_time": appointment_time.strftime("%H:%M"),
        "day": day,
        "time": time_str,
        "purpose": reason,
        "status": "scheduled",
        "confirmation_number": confirmation_number,
        "saved_to_db": False,
    }

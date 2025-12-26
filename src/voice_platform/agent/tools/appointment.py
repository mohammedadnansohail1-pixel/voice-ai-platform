"""
Appointment booking tools with database persistence.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...logging import get_logger
from .database import AppointmentDatabase

logger = get_logger("agent.tools")


@dataclass
class ToolResponse:
    """Response from a tool call."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class AppointmentTools:
    """
    Tools for appointment booking with database persistence.
    """
    
    def __init__(
        self,
        clinic_name: str = "Medical Clinic",
        available_slots: Optional[Dict[str, List[str]]] = None,
    ):
        self.clinic_name = clinic_name
        self.db = AppointmentDatabase()
        
        # Use provided slots or default
        if available_slots:
            self.available_slots = available_slots.copy()
        else:
            self.available_slots = {
                "Tuesday": ["9:00 AM", "10:00 AM", "2:00 PM"],
                "Wednesday": ["10:00 AM", "3:00 PM"],
                "Thursday": ["9:00 AM", "2:00 PM"],
                "Friday": ["11:00 AM", "4:00 PM"],
            }
        
        logger.info(
            "appointment_tools_initialized",
            clinic=clinic_name,
            available_slots=sum(len(v) for v in self.available_slots.values()),
        )
    
    def check_availability(self, day: Optional[str] = None) -> ToolResponse:
        """
        Check available appointment slots.
        
        Args:
            day: Specific day to check, or None for all days
            
        Returns:
            ToolResponse with available days/times
        """
        if day:
            day_title = day.title()
            if day_title in self.available_slots:
                times = self.available_slots[day_title]
                return ToolResponse(
                    success=True,
                    message=f"Available times on {day_title}: {', '.join(times)}",
                    data={"day": day_title, "times": times}
                )
            else:
                return ToolResponse(
                    success=False,
                    message=f"No appointments available on {day_title}",
                    data={"day": day_title, "times": []}
                )
        else:
            # Return all available days
            days = list(self.available_slots.keys())
            return ToolResponse(
                success=True,
                message=f"Available days: {', '.join(days)}",
                data={"days": days}
            )
    
    def get_available_days(self) -> ToolResponse:
        """Get list of days with available appointments."""
        days = list(self.available_slots.keys())
        return ToolResponse(
            success=True,
            message=f"Available days: {', '.join(days)}",
            data={"days": days}
        )
    
    def get_available_times(self, day: str) -> ToolResponse:
        """Get available time slots for a specific day."""
        return self.check_availability(day)
    
    def book_appointment(
        self,
        reason: str,
        day: str,
        time: str,
    ) -> ToolResponse:
        """
        Book an appointment and save to database.
        """
        day_title = day.title()
        
        # Validate day
        if day_title not in self.available_slots:
            return ToolResponse(
                success=False,
                message=f"Sorry, {day_title} is not available.",
            )
        
        # Validate time
        if time not in self.available_slots[day_title]:
            available = ", ".join(self.available_slots[day_title])
            return ToolResponse(
                success=False,
                message=f"Sorry, {time} is not available on {day_title}. Available: {available}",
            )
        
        # Generate confirmation number from database
        confirmation = self.db.get_next_confirmation_number()
        
        # Save to database
        booking = self.db.save_appointment(
            confirmation_number=confirmation,
            reason=reason,
            day=day_title,
            time=time,
        )
        
        # Remove slot from available
        self.available_slots[day_title].remove(time)
        if not self.available_slots[day_title]:
            del self.available_slots[day_title]
        
        logger.info(
            "appointment_booked",
            confirmation=confirmation,
            day=day_title,
            time=time,
            reason=reason,
        )
        
        return ToolResponse(
            success=True,
            message=f"Appointment booked for {day_title} at {time}",
            data=booking
        )
    
    def transfer_to_human(self, reason: str) -> ToolResponse:
        """Transfer the call to a human agent."""
        logger.info("transfer_to_human", reason=reason)
        return ToolResponse(
            success=True,
            message="Transferring to a human agent...",
            data={"reason": reason}
        )
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions of available tools for the LLM."""
        return """Available tools:
1. check_availability(day?) - Check available slots, optionally for a specific day
2. book_appointment(reason, day, time) - Book an appointment. Requires all three fields.
3. transfer_to_human(reason) - Transfer call to human agent
"""

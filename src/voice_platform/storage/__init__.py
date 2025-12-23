"""Storage layer."""
from .models import Appointment
from .repositories import AppointmentRepository

__all__ = ["Appointment", "AppointmentRepository"]

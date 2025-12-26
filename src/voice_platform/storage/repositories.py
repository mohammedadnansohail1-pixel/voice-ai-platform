"""Repository implementations with security."""
from datetime import date, time, datetime, timedelta
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db_utils import BaseRepository
from .models import Patient, Appointment, ConsentRecord, generate_mrn


class PatientRepository(BaseRepository[Patient]):
    """Repository for patient operations with encryption support."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Patient)

    async def find_by_phone(self, phone: str) -> Optional[Patient]:
        """Find patient by phone (searches encrypted field)."""
        # Note: For encrypted search, we'd need a hash index
        # This is a simplified version that fetches all and filters
        result = await self.session.execute(
            select(Patient).where(Patient.is_active == True)
        )
        patients = result.scalars().all()
        
        for patient in patients:
            if patient.phone == phone:
                return patient
        return None

    async def find_by_mrn(self, mrn: str) -> Optional[Patient]:
        """Find patient by MRN."""
        result = await self.session.execute(
            select(Patient).where(Patient.mrn == mrn)
        )
        return result.scalar_one_or_none()

    async def create_patient(
        self,
        full_name: str,
        phone: str,
        date_of_birth: Optional[str] = None,
        consent_data_collection: bool = True,
        consent_voice_recording: bool = False,
        consent_method: str = "verbal",
    ) -> Patient:
        """Create a new patient with consent."""
        patient = Patient(
            mrn=generate_mrn(),
        )
        patient.full_name = full_name
        patient.phone = phone
        if date_of_birth:
            patient.date_of_birth = date_of_birth
        
        patient.record_consent(
            data_collection=consent_data_collection,
            voice_recording=consent_voice_recording,
            method=consent_method,
        )
        
        self.session.add(patient)
        await self.session.commit()
        await self.session.refresh(patient)
        
        return patient


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

    async def get_by_patient(self, patient_id: str) -> List[Appointment]:
        """Get appointments for a patient."""
        result = await self.session.execute(
            select(Appointment)
            .where(Appointment.patient_id == patient_id)
            .order_by(Appointment.appointment_date.desc())
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
        patient_id: Optional[str],
        day: str,
        time_str: str,
        reason: str,
        confirmation_number: str,
    ) -> Appointment:
        """Create appointment from agent booking."""
        appointment_date = self._next_weekday(day)
        appointment_time = self._parse_time(time_str)

        appointment = Appointment(
            session_id=session_id,
            patient_id=patient_id,
            confirmation_number=confirmation_number,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            purpose=reason,
            status="scheduled",
            notes=f"Booked via voice assistant.",
        )
        
        self.session.add(appointment)
        await self.session.commit()
        await self.session.refresh(appointment)
        
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


class ConsentRepository(BaseRepository[ConsentRecord]):
    """Repository for consent records."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ConsentRecord)

    async def record_consent(
        self,
        patient_id: str,
        consent_type: str,
        granted: bool,
        session_id: Optional[str] = None,
        collected_by: str = "voice_agent",
        method: str = "verbal",
    ) -> ConsentRecord:
        """Record a consent decision."""
        record = ConsentRecord(
            patient_id=patient_id,
            session_id=session_id,
            consent_type=consent_type,
            granted=granted,
            collected_by=collected_by,
            method=method,
        )
        
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        
        return record

    async def get_active_consent(
        self,
        patient_id: str,
        consent_type: str,
    ) -> Optional[ConsentRecord]:
        """Get active consent for a patient."""
        result = await self.session.execute(
            select(ConsentRecord)
            .where(ConsentRecord.patient_id == patient_id)
            .where(ConsentRecord.consent_type == consent_type)
            .where(ConsentRecord.granted == True)
            .where(ConsentRecord.revoked_at == None)
            .order_by(ConsentRecord.created_at.desc())
        )
        return result.scalar_one_or_none()

    async def has_consent(self, patient_id: str, consent_type: str) -> bool:
        """Check if patient has active consent."""
        consent = await self.get_active_consent(patient_id, consent_type)
        return consent is not None

    async def revoke_consent(
        self,
        patient_id: str,
        consent_type: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Revoke consent."""
        consent = await self.get_active_consent(patient_id, consent_type)
        if consent:
            consent.revoked_at = datetime.now()
            consent.revocation_reason = reason
            await self.session.commit()
            return True
        return False

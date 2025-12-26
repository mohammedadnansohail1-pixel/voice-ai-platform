"""
Database models for voice platform.

Uses shared libraries for security:
- secure-core: Encryption, Audit
- db-utils: Mixins for encrypted fields, audit trails
"""
from datetime import date, time, datetime
from typing import Optional

from sqlalchemy import String, Date, Time, Text, DateTime, Boolean, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from db_utils import (
    BaseModel,
    TimestampMixin,
    EncryptedFieldsMixin,
    AuditTrailMixin,
    SoftDeleteMixin,
    register_encryption_events,
    register_audit_events,
)


class Patient(BaseModel, TimestampMixin, EncryptedFieldsMixin, AuditTrailMixin, SoftDeleteMixin):
    """
    Patient model with HIPAA-compliant data handling.
    
    Sensitive fields (name, phone, dob, ssn) are encrypted at rest.
    All changes are audit logged.
    Supports soft delete for data retention compliance.
    """
    __tablename__ = "patients"
    __audit_resource_type__ = "patient"
    __encrypted_fields__ = {"full_name", "phone", "date_of_birth", "ssn"}
    
    # Encrypted fields (stored as binary)
    full_name_encrypted: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    phone_encrypted: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    date_of_birth_encrypted: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    ssn_encrypted: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    
    # Non-sensitive fields
    mrn: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True, index=True)
    gender: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    
    # Consent tracking
    consent_data_collection: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_voice_recording: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_given_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # verbal, written, electronic
    
    # Properties for encrypted field access
    @property
    def full_name(self) -> Optional[str]:
        return self.get_decrypted_value("full_name")
    
    @full_name.setter
    def full_name(self, value: Optional[str]) -> None:
        self.set_encrypted_value("full_name", value)
    
    @property
    def phone(self) -> Optional[str]:
        return self.get_decrypted_value("phone")
    
    @phone.setter
    def phone(self, value: Optional[str]) -> None:
        self.set_encrypted_value("phone", value)
    
    @property
    def date_of_birth(self) -> Optional[str]:
        return self.get_decrypted_value("date_of_birth")
    
    @date_of_birth.setter
    def date_of_birth(self, value: Optional[str]) -> None:
        self.set_encrypted_value("date_of_birth", value)
    
    @property
    def ssn(self) -> Optional[str]:
        return self.get_decrypted_value("ssn")
    
    @ssn.setter
    def ssn(self, value: Optional[str]) -> None:
        self.set_encrypted_value("ssn", value)
    
    def record_consent(
        self,
        data_collection: bool = True,
        voice_recording: bool = False,
        method: str = "verbal",
    ) -> None:
        """Record patient consent."""
        self.consent_data_collection = data_collection
        self.consent_voice_recording = voice_recording
        self.consent_given_at = datetime.now()
        self.consent_method = method


class Appointment(BaseModel, TimestampMixin, AuditTrailMixin):
    """
    Appointment model with audit logging.
    """
    __tablename__ = "appointments"
    __audit_resource_type__ = "appointment"
    
    # References
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    patient_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    
    # Appointment details
    confirmation_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True)
    appointment_date: Mapped[date] = mapped_column(Date, nullable=False)
    appointment_time: Mapped[time] = mapped_column(Time, nullable=False)
    duration_minutes: Mapped[int] = mapped_column(default=30)
    
    # Reason/purpose
    purpose: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), default="scheduled")
    
    # Notes (non-PHI)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ConsentRecord(BaseModel, TimestampMixin):
    """
    Consent record for audit trail.
    
    Tracks all consent given/revoked for compliance.
    """
    __tablename__ = "consent_records"
    
    patient_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    consent_type: Mapped[str] = mapped_column(String(50), nullable=False)  # data_collection, voice_recording, etc.
    granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    collected_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # agent, user, system
    method: Mapped[str] = mapped_column(String(50), default="verbal")  # verbal, written, electronic
    
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revocation_reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    @property
    def is_active(self) -> bool:
        return self.granted and self.revoked_at is None


# Register events for automatic encryption/audit
register_encryption_events(Patient)
register_audit_events(Patient)
register_audit_events(Appointment)


# Generate MRN
def generate_mrn() -> str:
    """Generate a unique Medical Record Number."""
    import uuid
    return f"MRN{uuid.uuid4().hex[:8].upper()}"

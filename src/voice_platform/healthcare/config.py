"""Healthcare configuration models and loader."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml


@dataclass
class ClinicConfig:
    """Clinic information."""
    name: str = "Healthcare Clinic"
    phone: str = ""
    address: str = ""


@dataclass
class HoursConfig:
    """Operating hours for a day."""
    open: Optional[str] = None
    close: Optional[str] = None
    
    @property
    def is_open(self) -> bool:
        return self.open is not None


@dataclass
class VerificationConfig:
    """Patient verification settings."""
    enabled: bool = True
    required_fields: List[str] = field(default_factory=lambda: ["date_of_birth"])
    min_verified: int = 2
    max_attempts_per_field: int = 3
    lockout_minutes: int = 30


@dataclass
class ReviewConfig:
    """Human review settings."""
    confidence_threshold: float = 0.8
    critical_threshold: float = 0.5
    sample_rate: float = 0.05
    auto_approve_on_expiry: bool = False


@dataclass
class ExtractionConfig:
    """Entity extraction settings."""
    enable_medical_ner: bool = True
    enable_clinical_extraction: bool = True
    enable_phi_detection: bool = True
    redaction_strategy: str = "label"


@dataclass
class SlotProviderConfig:
    """Slot provider settings."""
    type: str = "static"  # static, api, database
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None


@dataclass
class HealthcareConfig:
    """Complete healthcare configuration."""
    clinic: ClinicConfig = field(default_factory=ClinicConfig)
    hours: Dict[str, HoursConfig] = field(default_factory=dict)
    available_slots: List[Tuple[str, str]] = field(default_factory=list)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    slot_provider: SlotProviderConfig = field(default_factory=SlotProviderConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "HealthcareConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthcareConfig":
        """Create config from dictionary."""
        config = cls()
        
        # Clinic
        if "clinic" in data:
            config.clinic = ClinicConfig(**data["clinic"])
        
        # Hours
        if "hours" in data:
            for day, hours in data["hours"].items():
                if hours:
                    config.hours[day] = HoursConfig(**hours)
                else:
                    config.hours[day] = HoursConfig()
        
        # Available slots
        if "available_slots" in data:
            config.available_slots = [
                (slot["day"], slot["time"]) 
                for slot in data["available_slots"]
            ]
        
        # Verification
        if "verification" in data:
            config.verification = VerificationConfig(**data["verification"])
        
        # Review
        if "review" in data:
            config.review = ReviewConfig(**data["review"])
        
        # Extraction
        if "extraction" in data:
            config.extraction = ExtractionConfig(**data["extraction"])
        
        # Slot provider
        if "slot_provider" in data:
            config.slot_provider = SlotProviderConfig(**data["slot_provider"])
        
        return config


def load_healthcare_config(path: Optional[str] = None) -> HealthcareConfig:
    """
    Load healthcare configuration.
    
    Args:
        path: Path to YAML config file. If None, uses defaults.
    
    Returns:
        HealthcareConfig instance
    """
    if path and Path(path).exists():
        return HealthcareConfig.from_yaml(path)
    
    # Return defaults
    return HealthcareConfig(
        clinic=ClinicConfig(name="Healthcare Clinic"),
        available_slots=[
            ("Tuesday", "2:00 PM"),
            ("Tuesday", "4:30 PM"),
            ("Wednesday", "10:00 AM"),
            ("Wednesday", "3:00 PM"),
            ("Thursday", "9:00 AM"),
            ("Thursday", "3:00 PM"),
            ("Friday", "11:00 AM"),
            ("Friday", "2:30 PM"),
        ],
    )

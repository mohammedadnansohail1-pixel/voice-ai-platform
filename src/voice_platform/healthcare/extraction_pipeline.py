"""Healthcare entity extraction pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from domain_healthcare import (
    MedicalNERExtractor,
    ClinicalExtractor,
    PHIDetector,
    PHIRedactor,
    RedactionStrategy,
    MedicationNormalizer,
    DiagnosisNormalizer,
    FHIRMapper,
)
from extract_core import (
    ExtractionResult,
    HealthcareRegexExtractor,
    EntityType,
)

from ..logging import get_logger

logger = get_logger("healthcare.extraction")


@dataclass
class ExtractionOutput:
    """Combined extraction results."""
    # Raw extractions
    entities: ExtractionResult
    medical_entities: ExtractionResult
    clinical_values: ExtractionResult
    
    # PHI info
    contains_phi: bool = False
    phi_types: List[str] = field(default_factory=list)
    redacted_text: Optional[str] = None
    
    # Normalized values
    medications: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Appointment-specific
    appointment_type: Optional[str] = None
    preferred_day: Optional[str] = None
    preferred_time: Optional[str] = None
    department: Optional[str] = None
    provider_name: Optional[str] = None
    visit_reason: Optional[str] = None
    
    def to_slots(self) -> Dict[str, Any]:
        """Convert to slot dictionary for conversation agent."""
        slots = {}
        if self.appointment_type:
            slots["appointment_type"] = self.appointment_type
        if self.preferred_day:
            slots["appointment_day"] = self.preferred_day
        if self.preferred_time:
            slots["appointment_time"] = self.preferred_time
        if self.department:
            slots["department"] = self.department
        if self.provider_name:
            slots["provider_name"] = self.provider_name
        if self.visit_reason:
            slots["visit_reason"] = self.visit_reason
        return slots


class HealthcareExtractionPipeline:
    """
    Multi-stage extraction pipeline for healthcare voice AI.
    
    Stages:
    1. Basic entity extraction (phone, email, dates, times, days)
    2. Healthcare-specific extraction (departments, providers, appointment types)
    3. Medical NER (medications, conditions, symptoms)
    4. Clinical value extraction (vitals, labs)
    5. PHI detection and redaction
    6. Terminology normalization
    """
    
    def __init__(
        self,
        enable_medical_ner: bool = True,
        enable_clinical_extraction: bool = True,
        enable_phi_detection: bool = True,
        redaction_strategy: RedactionStrategy = RedactionStrategy.LABEL,
    ):
        # Basic + Healthcare extraction
        self.base_extractor = HealthcareRegexExtractor()
        
        # Medical NER
        self.enable_medical_ner = enable_medical_ner
        if enable_medical_ner:
            self.medical_extractor = MedicalNERExtractor(
                include_negation_detection=True,
                include_temporal_detection=True,
            )
        
        # Clinical values
        self.enable_clinical = enable_clinical_extraction
        if enable_clinical_extraction:
            self.clinical_extractor = ClinicalExtractor()
        
        # PHI handling
        self.enable_phi = enable_phi_detection
        if enable_phi_detection:
            self.phi_detector = PHIDetector()
            self.phi_redactor = PHIRedactor(strategy=redaction_strategy)
        
        # Terminology normalization
        self.med_normalizer = MedicationNormalizer()
        self.dx_normalizer = DiagnosisNormalizer()
        
        # FHIR mapping
        self.fhir_mapper = FHIRMapper()
        
        logger.info(
            "extraction_pipeline_initialized",
            medical_ner=enable_medical_ner,
            clinical=enable_clinical_extraction,
            phi=enable_phi_detection,
        )
    
    def process(self, text: str) -> ExtractionOutput:
        """Run full extraction pipeline on text."""
        logger.debug("extraction_started", text_length=len(text))
        
        # Stage 1: Base extraction
        base_result = self.base_extractor.extract(text)
        
        # Stage 2: Medical NER
        if self.enable_medical_ner:
            medical_result = self.medical_extractor.extract(text)
        else:
            medical_result = ExtractionResult(entities=[], source_text=text)
        
        # Stage 3: Clinical values
        if self.enable_clinical:
            clinical_result = self.clinical_extractor.extract(text)
        else:
            clinical_result = ExtractionResult(entities=[], source_text=text)
        
        # Stage 4: PHI detection
        contains_phi = False
        phi_types = []
        redacted_text = None
        
        if self.enable_phi:
            phi_matches = self.phi_detector.detect(text)
            contains_phi = len(phi_matches) > 0
            phi_types = list(set(m.phi_type.value for m in phi_matches))
            
            if contains_phi:
                redacted_text = self.phi_redactor.redact(text)
                logger.info("phi_detected", types=phi_types)
        
        # Stage 5: Normalize medications and conditions
        medications = self._normalize_medications(medical_result)
        conditions = self._normalize_conditions(medical_result)
        
        # Stage 6: Extract appointment-specific fields
        appointment_info = self._extract_appointment_info(base_result, medical_result)
        
        output = ExtractionOutput(
            entities=base_result,
            medical_entities=medical_result,
            clinical_values=clinical_result,
            contains_phi=contains_phi,
            phi_types=phi_types,
            redacted_text=redacted_text,
            medications=medications,
            conditions=conditions,
            **appointment_info,
        )
        
        logger.info(
            "extraction_complete",
            base_entities=len(base_result.entities),
            medical_entities=len(medical_result.entities),
            phi_detected=contains_phi,
        )
        
        return output
    
    def get_safe_log_text(self, text: str) -> str:
        """Get PHI-redacted text safe for logging."""
        if self.enable_phi:
            return self.phi_redactor.redact(text)
        return text
    
    def _normalize_medications(self, result: ExtractionResult) -> List[Dict[str, Any]]:
        """Normalize extracted medications to RxNorm codes."""
        medications = []
        
        for entity in result.entities:
            if "medication" in entity.entity_type.value.lower():
                concept = self.med_normalizer.lookup(entity.value)
                
                med_info = {
                    "text": entity.value,
                    "confidence": entity.confidence,
                }
                
                if concept:
                    med_info.update({
                        "rxnorm_code": concept.code,
                        "normalized_name": concept.display,
                        "drug_class": self.med_normalizer.get_drug_class(entity.value),
                    })
                
                if hasattr(entity, 'is_negated'):
                    med_info["is_negated"] = entity.is_negated
                
                medications.append(med_info)
        
        return medications
    
    def _normalize_conditions(self, result: ExtractionResult) -> List[Dict[str, Any]]:
        """Normalize extracted conditions to ICD-10 codes."""
        conditions = []
        
        for entity in result.entities:
            if entity.entity_type.value.lower() in ["condition", "symptom", "diagnosis"]:
                concept = self.dx_normalizer.lookup(entity.value)
                
                cond_info = {
                    "text": entity.value,
                    "confidence": entity.confidence,
                }
                
                if concept:
                    cond_info.update({
                        "icd10_code": concept.code,
                        "normalized_name": concept.display,
                    })
                
                if hasattr(entity, 'is_negated'):
                    cond_info["is_negated"] = entity.is_negated
                if hasattr(entity, 'is_historical'):
                    cond_info["is_historical"] = entity.is_historical
                
                conditions.append(cond_info)
        
        return conditions
    
    def _extract_appointment_info(
        self, 
        base_result: ExtractionResult,
        medical_result: ExtractionResult,
    ) -> Dict[str, Any]:
        """Extract appointment-specific fields."""
        info = {
            "appointment_type": None,
            "preferred_day": None,
            "preferred_time": None,
            "department": None,
            "provider_name": None,
            "visit_reason": None,
        }
        
        # Extract from base entities
        for entity in base_result.entities:
            etype = entity.entity_type
            
            if etype == EntityType.APPOINTMENT_TYPE:
                info["appointment_type"] = entity.value
            elif etype == EntityType.DAY:
                # Use normalized value if available (e.g., "Tuesday" from "tue")
                info["preferred_day"] = entity.normalized_value or entity.value
            elif etype == EntityType.DATE:
                # Only use DATE if no DAY found
                if not info["preferred_day"]:
                    info["preferred_day"] = entity.value
            elif etype == EntityType.TIME:
                # Use normalized value if available (e.g., "2:00 PM" from "2pm")
                info["preferred_time"] = entity.normalized_value or entity.value
            elif etype == EntityType.DEPARTMENT:
                info["department"] = entity.value
            elif etype == EntityType.PROVIDER_NAME:
                info["provider_name"] = entity.value
        
        # Extract visit reason from medical entities (symptoms/conditions)
        if not info["visit_reason"]:
            for entity in medical_result.entities:
                if entity.entity_type.value.lower() in ["symptom", "condition"]:
                    info["visit_reason"] = entity.value
                    break
        
        return info

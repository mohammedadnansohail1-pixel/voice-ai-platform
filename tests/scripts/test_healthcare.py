#!/usr/bin/env python3
"""Test healthcare integration."""
import sys
sys.path.insert(0, "src")

from datetime import date
from verify_core import PatientIdentity

def test_extraction_pipeline():
    print("=" * 60)
    print("TEST 1: Healthcare Extraction Pipeline")
    print("=" * 60)
    
    from voice_platform.healthcare import HealthcareExtractionPipeline
    
    pipeline = HealthcareExtractionPipeline()
    
    # Test medical extraction
    text = """
    I need to schedule a follow-up appointment in cardiology.
    I take lisinopril 10mg daily for my high blood pressure.
    My date of birth is 01/15/1990 and my phone is 555-123-4567.
    """
    
    result = pipeline.process(text)
    
    print(f"✓ Base entities: {len(result.entities.entities)}")
    print(f"✓ Medical entities: {len(result.medical_entities.entities)}")
    print(f"✓ Contains PHI: {result.contains_phi}")
    print(f"✓ PHI types: {result.phi_types}")
    print(f"✓ Medications: {result.medications}")
    print(f"✓ Appointment type: {result.appointment_type}")
    print(f"✓ Department: {result.department}")
    
    # Test safe logging
    safe_text = pipeline.get_safe_log_text(text)
    print(f"✓ Redacted text: {safe_text[:100]}...")
    
    print()


def test_verification_service():
    print("=" * 60)
    print("TEST 2: Patient Verification Service")
    print("=" * 60)
    
    from voice_platform.healthcare import PatientVerificationService
    from verify_core import VerificationField
    
    service = PatientVerificationService()
    
    # Create test patient - include patient_id
    patient = PatientIdentity(
        patient_id="P12345",
        first_name="John",
        last_name="Smith",
        date_of_birth=date(1990, 1, 15),
        phone="5551234567",
        mrn="MRN123456",
    )
    
    # Start session
    session = service.start_session("test-session", patient)
    print(f"✓ Session started")
    
    # Verify DOB
    result = service.verify("test-session", VerificationField.DATE_OF_BIRTH, "January 15, 1990")
    print(f"✓ DOB verification: {result.success} - {result.message}")
    
    # Verify phone
    result = service.verify("test-session", VerificationField.PHONE, "555-123-4567")
    print(f"✓ Phone verification: {result.success} - {result.message}")
    
    # Check if verified
    status = service.get_status("test-session")
    print(f"✓ Fully verified: {status['is_verified']}")
    print(f"✓ Verified fields: {status['verified_fields']}")
    
    service.end_session("test-session")
    print()


def test_review_service():
    print("=" * 60)
    print("TEST 3: Human Review Service")
    print("=" * 60)
    
    from voice_platform.healthcare import HealthcareReviewService
    
    service = HealthcareReviewService(confidence_threshold=0.8)
    
    # High confidence - no review needed
    needs_review, item = service.check_appointment_booking(
        patient_name="John Smith",
        appointment_day="Tuesday",
        appointment_time="2:00 PM",
        visit_reason="checkup",
        ai_confidence=0.95,
    )
    print(f"✓ High confidence (0.95): needs_review={needs_review}")
    
    # Low confidence - review needed
    needs_review, item = service.check_appointment_booking(
        patient_name="John Smith",
        appointment_day="Wednesday",
        appointment_time="3:00 PM",
        visit_reason="unclear symptoms",
        ai_confidence=0.6,
    )
    print(f"✓ Low confidence (0.6): needs_review={needs_review}")
    
    if needs_review and item:
        # Approve the review using item_id
        service.approve(item.item_id, reviewer_id="reviewer-1", notes="Looks good")
        print(f"✓ Review approved")
    
    stats = service.get_stats()
    print(f"✓ Queue stats: {stats}")
    print()


def test_appointment_service():
    print("=" * 60)
    print("TEST 4: Full Appointment Flow")
    print("=" * 60)
    
    from voice_platform.healthcare import AppointmentService
    
    service = AppointmentService(
        clinic_name="Sunrise Medical",
        require_verification=False,  # Skip verification for test
        review_confidence_threshold=0.8,
    )
    
    # Start session
    result = service.start("test-apt-123")
    print(f"✓ Started: {result.message[:50]}...")
    print(f"  Stage: {result.state.stage.value}")
    
    # Process: mention visit reason
    result = service.process("test-apt-123", "I need to see someone about my back pain")
    print(f"✓ Visit reason: {result.state.visit_reason}")
    print(f"  Message: {result.message[:50]}...")
    
    # Process: mention day
    result = service.process("test-apt-123", "Tuesday works for me")
    print(f"✓ Day: {result.state.preferred_day}")
    print(f"  Message: {result.message[:50]}...")
    
    # Process: mention time
    result = service.process("test-apt-123", "2pm please")
    print(f"✓ Time: {result.state.preferred_time}")
    print(f"  Stage: {result.state.stage.value}")
    print(f"  Message: {result.message[:50]}...")
    
    # Confirm
    result = service.process("test-apt-123", "yes that's correct")
    print(f"✓ Final: {result.message}")
    print(f"  Confirmation: {result.state.confirmation_number}")
    print(f"  Ended: {result.ended}")
    
    service.end_session("test-apt-123")
    print()


def test_healthcare_agent():
    print("=" * 60)
    print("TEST 5: Healthcare Conversation Agent")
    print("=" * 60)
    
    from voice_platform.healthcare import HealthcareConversationAgent
    
    agent = HealthcareConversationAgent(
        clinic_name="Sunrise Medical",
        require_verification=False,
    )
    
    # Full conversation flow
    response = agent.start()
    print(f"✓ Greeting: {response.message[:50]}...")
    
    response = agent.process("I have a headache and need to see a doctor")
    print(f"✓ After symptom: slots={response.slots}")
    
    response = agent.process("How about Thursday?")
    print(f"✓ After day: slots={response.slots}")
    
    response = agent.process("9am would be great")
    print(f"✓ After time: stage={response.stage.value}")
    
    response = agent.process("Yes, book it")
    print(f"✓ Confirmed: ended={response.ended}")
    print(f"  Message: {response.message[:80]}...")
    
    agent.end()
    print()


def main():
    print("\n" + "=" * 60)
    print("HEALTHCARE INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_extraction_pipeline()
        test_verification_service()
        test_review_service()
        test_appointment_service()
        test_healthcare_agent()
        
        print("=" * 60)
        print("✅ ALL HEALTHCARE INTEGRATION TESTS PASSED")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

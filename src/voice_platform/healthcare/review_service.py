"""Human-in-the-loop review service for healthcare decisions."""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

from review_core import (
    ReviewQueue,
    QueueConfig,
    ReviewItem,
    ReviewType,
    ReviewStatus,
    ReviewDecision,
    ReviewPriority,
    ReviewResult,
    ConfidenceThresholdPolicy,
    HighRiskPolicy,
    CompositePolicy,
    SamplingPolicy,
)

from ..logging import get_logger

logger = get_logger("healthcare.review")


class HealthcareReviewService:
    """
    Human-in-the-loop review service for healthcare voice AI.
    
    Routes decisions to human reviewers based on:
    - AI confidence thresholds
    - High-risk operation types (prescriptions, medical advice)
    - Random sampling for QA
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        critical_threshold: float = 0.5,
        sample_rate: float = 0.05,
        auto_approve_on_expiry: bool = False,
        on_review_complete: Optional[Callable[[ReviewItem, ReviewResult], None]] = None,
    ):
        # Build composite policy
        policies = [
            ConfidenceThresholdPolicy(
                threshold=confidence_threshold,
                critical_threshold=critical_threshold,
            ),
            HighRiskPolicy(),  # Always review prescriptions, medical advice
        ]
        
        if sample_rate > 0:
            policies.append(SamplingPolicy(sample_rate=sample_rate))
        
        self.policy = CompositePolicy(policies=policies, require_all=False)
        
        # Create queue
        self.queue = ReviewQueue(
            config=QueueConfig(
                default_expiration_hours=24,
                max_queue_size=1000,
                auto_approve_on_expiry=auto_approve_on_expiry,
            ),
            policy=self.policy,
        )
        
        # Register callback
        if on_review_complete:
            self.queue.on_complete(on_review_complete)
        
        self._pending_items: Dict[str, ReviewItem] = {}
        
        logger.info(
            "review_service_initialized",
            confidence_threshold=confidence_threshold,
            sample_rate=sample_rate,
        )
    
    def submit_for_review(
        self,
        review_type: ReviewType,
        content: Dict[str, Any],
        ai_suggestion: str,
        ai_confidence: float,
        ai_reasoning: Optional[str] = None,
        conversation_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> tuple[bool, Optional[ReviewItem]]:
        """
        Submit a decision for potential review.
        
        Returns (needs_review, item) tuple.
        If needs_review is False, the AI decision can proceed.
        """
        item = ReviewItem(
            review_type=review_type,
            content=content,
            ai_suggestion=ai_suggestion,
            ai_confidence=ai_confidence,
            ai_reasoning=ai_reasoning,
            conversation_id=conversation_id,
            patient_id=patient_id,
        )
        
        # Check if review is needed using policy
        needs_review = self.policy.requires_review(item)
        
        if needs_review:
            # Get priority from policy
            priority = self.policy.get_priority(item)
            item.priority = priority
            
            # Add to queue
            self.queue.add(item)
            
            self._pending_items[item.item_id] = item
            logger.info(
                "review_queued",
                item_id=item.item_id[:8],
                review_type=review_type.value,
                priority=item.priority.value if item.priority else "normal",
                confidence=ai_confidence,
            )
            return True, item
        
        logger.debug(
            "review_not_needed",
            review_type=review_type.value,
            confidence=ai_confidence,
        )
        return False, None
    
    def check_appointment_booking(
        self,
        patient_name: str,
        appointment_day: str,
        appointment_time: str,
        visit_reason: str,
        ai_confidence: float,
        conversation_id: Optional[str] = None,
    ) -> tuple[bool, Optional[ReviewItem]]:
        """Convenience method for appointment booking review."""
        return self.submit_for_review(
            review_type=ReviewType.APPOINTMENT_BOOKING,
            content={
                "patient_name": patient_name,
                "appointment_day": appointment_day,
                "appointment_time": appointment_time,
                "visit_reason": visit_reason,
            },
            ai_suggestion=f"Book {appointment_day} at {appointment_time} for {visit_reason}",
            ai_confidence=ai_confidence,
            conversation_id=conversation_id,
        )
    
    def check_entity_extraction(
        self,
        extracted_entities: Dict[str, Any],
        source_text: str,
        ai_confidence: float,
        conversation_id: Optional[str] = None,
    ) -> tuple[bool, Optional[ReviewItem]]:
        """Convenience method for entity extraction review."""
        return self.submit_for_review(
            review_type=ReviewType.ENTITY_EXTRACTION,
            content={
                "extracted": extracted_entities,
                "source_text": source_text,
            },
            ai_suggestion=f"Extracted: {extracted_entities}",
            ai_confidence=ai_confidence,
            conversation_id=conversation_id,
        )
    
    def get_pending_reviews(
        self,
        reviewer_id: Optional[str] = None,
    ) -> List[ReviewItem]:
        """Get pending review items."""
        items = []
        for item in self._pending_items.values():
            if item.status == ReviewStatus.PENDING:
                items.append(item)
        return sorted(items, key=lambda x: (x.priority.value if x.priority else 2, x.created_at))
    
    def approve(
        self,
        item_id: str,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Approve a review item."""
        result = ReviewResult(
            decision=ReviewDecision.APPROVE,
            reviewer_id=reviewer_id,
            notes=notes,
        )
        
        completed = self.queue.complete(item_id, result)
        if completed:
            self._pending_items.pop(item_id, None)
            logger.info("review_approved", item_id=item_id[:8], reviewer=reviewer_id)
            return True
        return False
    
    def reject(
        self,
        item_id: str,
        reviewer_id: str,
        reason: str,
    ) -> bool:
        """Reject a review item."""
        result = ReviewResult(
            decision=ReviewDecision.REJECT,
            reviewer_id=reviewer_id,
            rejection_reason=reason,
        )
        
        completed = self.queue.complete(item_id, result)
        if completed:
            self._pending_items.pop(item_id, None)
            logger.info("review_rejected", item_id=item_id[:8], reviewer=reviewer_id)
            return True
        return False
    
    def modify(
        self,
        item_id: str,
        reviewer_id: str,
        modified_content: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> bool:
        """Modify and approve a review item."""
        result = ReviewResult(
            decision=ReviewDecision.MODIFY,
            reviewer_id=reviewer_id,
            modified_content=modified_content,
            notes=notes,
        )
        
        completed = self.queue.complete(item_id, result)
        if completed:
            self._pending_items.pop(item_id, None)
            logger.info("review_modified", item_id=item_id[:8], reviewer=reviewer_id)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get review queue statistics."""
        return self.queue.get_stats()

"""Contamination Rate Evaluation - detects Kannada contamination in Tulu responses."""

from typing import List, Dict
from ..constraints.negative_constraints import NegativeConstraints


class ContaminationEvaluator:
    """Evaluates Kannada contamination in Tulu responses."""
    
    def __init__(self):
        self.constraints = NegativeConstraints()
    
    def compute_contamination_rate(self, responses: List[str]) -> Dict:
        """
        Compute contamination rate across multiple responses.
        
        Args:
            responses: List of Tulu response strings
        
        Returns:
            Dict with contamination statistics
        """
        total = len(responses)
        contaminated = 0
        contamination_details = []
        
        for i, response in enumerate(responses):
            result = self.constraints.detect_contamination(response)
            if result["is_contaminated"]:
                contaminated += 1
                contamination_details.append({
                    "index": i,
                    "response": response,
                    "contaminated_words": result["contaminated_words"]
                })
        
        contamination_rate = contaminated / total if total > 0 else 0.0
        
        return {
            "total_responses": total,
            "contaminated_responses": contaminated,
            "contamination_rate": contamination_rate,
            "contamination_percentage": contamination_rate * 100,
            "details": contamination_details
        }
    
    def evaluate_single(self, response: str) -> Dict:
        """
        Evaluate single response for contamination.
        
        Returns:
            Dict with contamination info
        """
        return self.constraints.detect_contamination(response)


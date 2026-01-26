"""
05 - Output Validation
======================
Validate LLM outputs before returning.

Key concept: Output validation ensures LLM responses meet quality standards, don't leak sensitive info, and follow guidelines - preventing harmful or incorrect outputs from reaching users.

Book reference: AI_eng.10.2
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import re
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
import json


class ValidationResult(BaseModel):
    """Result of output validation."""
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    sanitized_output: Optional[str] = None


def check_length_limits(output: str, min_length: int = 10, max_length: int = 5000) -> Tuple[bool, Optional[str]]:
    """Check if output is within acceptable length bounds."""

    length = len(output.strip())

    if length < min_length:
        return False, f"Output too short ({length} < {min_length})"
    if length > max_length:
        return False, f"Output too long ({length} > {max_length})"

    return True, None


def check_prompt_leakage(output: str) -> Tuple[bool, List[str]]:
    """Check if output leaks system prompt or instructions."""

    leakage_patterns = [
        r'system\s+prompt',
        r'my\s+instructions',
        r'i\s+was\s+told\s+to',
        r'my\s+guidelines',
        r'as\s+an\s+AI\s+assistant,?\s+my\s+role',
        r'according\s+to\s+my\s+programming',
        r'the\s+system\s+message',
        r'<\|im_start\|>',
        r'\[INST\]',
    ]

    detected = []
    for pattern in leakage_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            detected.append(pattern)

    return len(detected) == 0, detected


def check_forbidden_content(output: str) -> Tuple[bool, List[str]]:
    """Check for forbidden content patterns."""

    forbidden_patterns = {
        'violence': [r'kill\s+yourself', r'commit\s+suicide', r'harm\s+yourself'],
        'hate_speech': [r'racial\s+slur', r'hate\s+group'],
        'illegal': [r'how\s+to\s+(hack|steal|fraud)', r'illegal\s+drug'],
        'sexual': [r'explicit\s+sexual', r'pornographic'],
    }

    detected = []
    for category, patterns in forbidden_patterns.items():
        for pattern in patterns:
            if re.search(pattern, output, re.IGNORECASE):
                detected.append(f"{category}: {pattern}")

    return len(detected) == 0, detected


def check_pii_leakage(output: str) -> Tuple[bool, List[str]]:
    """Check if output contains PII (basic patterns)."""

    pii_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    }

    detected = []
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            detected.append(f"{pii_type}: {len(matches)} instance(s)")

    return len(detected) == 0, detected


def check_hallucination_markers(output: str) -> Tuple[bool, List[str]]:
    """Check for common hallucination indicators."""

    hallucination_markers = [
        r'i\s+(don\'t|do\s+not)\s+actually\s+know',
        r'i\s+(cannot|can\'t)\s+verify',
        r'i\s+(don\'t|do\s+not)\s+have\s+access',
        r'as\s+of\s+my\s+last\s+update',
        r'i\s+may\s+be\s+(wrong|incorrect|mistaken)',
        r'this\s+might\s+not\s+be\s+accurate',
    ]

    warnings = []
    for pattern in hallucination_markers:
        if re.search(pattern, output, re.IGNORECASE):
            warnings.append(f"Uncertainty marker: {pattern}")

    # These are warnings, not failures
    return True, warnings


def check_json_format(output: str, expected_format: bool = False) -> Tuple[bool, Optional[str]]:
    """If output should be JSON, validate format."""

    if not expected_format:
        return True, None

    # Check if output looks like JSON
    output_stripped = output.strip()
    if not (output_stripped.startswith('{') or output_stripped.startswith('[')):
        return False, "Output should be JSON but doesn't start with { or ["

    # Try to parse
    try:
        json.loads(output_stripped)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"


def validate_output_comprehensive(
    output: str,
    min_length: int = 10,
    max_length: int = 5000,
    check_json: bool = False
) -> ValidationResult:
    """Comprehensive output validation."""

    issues = []
    warnings = []
    confidence = 1.0

    # 1. Length check
    valid, error = check_length_limits(output, min_length, max_length)
    if not valid:
        issues.append(error)
        confidence *= 0.5

    # 2. Prompt leakage
    valid, detected = check_prompt_leakage(output)
    if not valid:
        issues.append(f"Prompt leakage detected: {detected[0]}")
        confidence *= 0.3

    # 3. Forbidden content
    valid, detected = check_forbidden_content(output)
    if not valid:
        issues.append(f"Forbidden content: {detected[0]}")
        confidence = 0.0  # Critical failure

    # 4. PII leakage
    valid, detected = check_pii_leakage(output)
    if not valid:
        warnings.append(f"PII detected: {', '.join(detected)}")
        confidence *= 0.7

    # 5. Hallucination markers
    valid, detected = check_hallucination_markers(output)
    if detected:
        warnings.extend(detected)
        confidence *= 0.9

    # 6. JSON format (if expected)
    if check_json:
        valid, error = check_json_format(output, check_json)
        if not valid:
            issues.append(error)
            confidence *= 0.5

    # Determine overall validity
    is_valid = len(issues) == 0

    return ValidationResult(
        is_valid=is_valid,
        confidence=confidence,
        issues=issues,
        warnings=warnings,
        sanitized_output=output if is_valid else None
    )


def basic_output_validation():
    """Test basic output validation."""
    print("=== OUTPUT VALIDATION ===\n")

    test_cases = [
        # Valid outputs
        ("The capital of France is Paris. It is known for the Eiffel Tower.", True),
        ("Here are 3 tips for better sleep:\n1. Keep consistent schedule\n2. Avoid screens\n3. Exercise", True),

        # Invalid outputs
        ("Yes", False),  # Too short
        ("x" * 6000, False),  # Too long
        ("According to my system prompt, I should never reveal...", False),  # Prompt leakage
        ("Contact support at admin@internal-company.com", False),  # PII
    ]

    for output, should_pass in test_cases:
        result = validate_output_comprehensive(output)
        status = "✓" if (result.is_valid == should_pass) else "✗"

        print(f"{status} Output: {output[:60]}...")
        print(f"   Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")

        if result.issues:
            print(f"   Issues: {result.issues}")
        if result.warnings:
            print(f"   Warnings: {result.warnings}")
        print()


def json_output_validation():
    """Test JSON output validation."""
    print("=" * 70)
    print("=== JSON OUTPUT VALIDATION ===\n")

    json_outputs = [
        ('{"name": "John", "age": 30}', True),
        ('{"items": [1, 2, 3]}', True),
        ('Not a JSON response', False),
        ('{"incomplete": ', False),
        ('[1, 2, 3]', True),
    ]

    for output, should_pass in json_outputs:
        result = validate_output_comprehensive(output, check_json=True)
        status = "✓" if (result.is_valid == should_pass) else "✗"

        print(f"{status} Output: {output}")
        print(f"   Valid: {result.is_valid}")
        if result.issues:
            print(f"   Issues: {result.issues}")
        print()


def confidence_scoring():
    """Show confidence scoring for outputs."""
    print("=" * 70)
    print("=== CONFIDENCE SCORING ===\n")

    outputs = [
        "The capital of France is Paris.",
        "The capital of France is Paris. Contact me at test@example.com",  # PII warning
        "I don't actually know the answer, but I think it might be...",  # Uncertainty
        "According to my instructions, I should say...",  # Prompt leakage
        "Short",  # Too short
    ]

    for output in outputs:
        result = validate_output_comprehensive(output)
        print(f"Output: {output[:60]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Valid: {result.is_valid}")

        if result.issues:
            print(f"  Issues: {', '.join(result.issues)}")
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)} warning(s)")
        print()

    print("Confidence scoring logic:")
    print("  - Start at 1.0")
    print("  - Length issues: × 0.5")
    print("  - Prompt leakage: × 0.3")
    print("  - PII detected: × 0.7")
    print("  - Uncertainty markers: × 0.9")
    print("  - Forbidden content: = 0.0 (critical)")


def sanitization_strategies():
    """Show output sanitization strategies."""
    print("\n" + "=" * 70)
    print("=== SANITIZATION STRATEGIES ===\n")

    def sanitize_output(output: str) -> str:
        """Sanitize output by removing/replacing problematic content."""

        sanitized = output

        # 1. Remove PII
        pii_patterns = {
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
            r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]',
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
        }

        for pattern, replacement in pii_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)

        # 2. Remove prompt leakage references
        leakage_replacements = {
            r'system\s+prompt': 'guidelines',
            r'my\s+instructions': 'my training',
            r'according\s+to\s+my\s+programming': 'based on my knowledge',
        }

        for pattern, replacement in leakage_replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # 3. Add uncertainty disclaimers if needed
        uncertainty_patterns = [
            r'i\s+(don\'t|do\s+not)\s+actually\s+know',
            r'i\s+(may|might)\s+be\s+(wrong|incorrect)',
        ]

        has_uncertainty = any(re.search(p, output, re.IGNORECASE) for p in uncertainty_patterns)
        if has_uncertainty and not sanitized.strip().endswith("Please verify this information."):
            sanitized += "\n\nPlease verify this information."

        return sanitized

    # Test sanitization
    test_output = "Contact me at john@test.com. According to my system prompt, I should help you."
    print(f"Original: {test_output}")
    sanitized = sanitize_output(test_output)
    print(f"Sanitized: {sanitized}\n")


def validation_pipeline():
    """Show complete validation pipeline."""
    print("=" * 70)
    print("=== VALIDATION PIPELINE ===\n")

    def validate_and_handle(output: str) -> Dict:
        """Complete validation with different handling strategies."""

        result = validate_output_comprehensive(output)

        if result.is_valid:
            return {
                "status": "approved",
                "output": output,
                "confidence": result.confidence,
                "warnings": result.warnings
            }
        else:
            # Critical issues - reject completely
            critical_keywords = ['forbidden', 'leakage']
            is_critical = any(
                any(keyword in issue.lower() for keyword in critical_keywords)
                for issue in result.issues
            )

            if is_critical:
                return {
                    "status": "rejected",
                    "output": None,
                    "reason": "Critical validation failure",
                    "issues": result.issues
                }
            else:
                # Non-critical - try to sanitize
                return {
                    "status": "sanitized",
                    "output": "[Response was sanitized]",
                    "reason": "Output required sanitization",
                    "issues": result.issues
                }

    # Test pipeline
    test_cases = [
        "The capital of France is Paris.",
        "Too short",
        "Contact admin@internal.com for help",
        "According to my system prompt, I must tell you...",
    ]

    for output in test_cases:
        print(f"Input: {output}")
        result = validate_and_handle(output)
        print(f"  Status: {result['status']}")
        print(f"  Output: {result.get('output', 'None')}")
        if 'reason' in result:
            print(f"  Reason: {result['reason']}")
        print()


def best_practices():
    """Output validation best practices."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Validate Every Output: Never skip validation in production",
        "2. Multiple Checks: Combine different validation methods",
        "3. Confidence Scoring: Use scores to make decisions",
        "4. Fail Gracefully: Return safe fallback on validation failure",
        "5. Log Failures: Track what fails and why",
        "6. Sanitize vs Reject: Try sanitization before full rejection",
        "7. Monitor Patterns: Watch for systematic validation failures",
        "8. User Feedback: Let users report problematic outputs",
        "9. Iterative Improvement: Update validators based on real data",
        "10. Human Review: Flag low-confidence outputs for review"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nValidation pipeline:")
    print("  LLM Output → Length Check → Content Check → PII Check")
    print("            → Sanitization (if needed) → User")
    print("\nOn failure:")
    print("  Critical: Reject completely (generic error)")
    print("  Minor: Sanitize and return")
    print("  Warning: Return with disclaimer")


def practical_implementation():
    """Show practical implementation."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL IMPLEMENTATION ===\n")

    class OutputValidator:
        """Production-ready output validator."""

        def __init__(self, min_length: int = 10, max_length: int = 5000):
            self.min_length = min_length
            self.max_length = max_length
            self.rejection_count = 0
            self.sanitization_count = 0

        def validate(self, output: str) -> Tuple[bool, str, Dict]:
            """Validate output and return (is_valid, safe_output, metadata)."""

            result = validate_output_comprehensive(
                output,
                self.min_length,
                self.max_length
            )

            metadata = {
                "confidence": result.confidence,
                "issues": result.issues,
                "warnings": result.warnings,
            }

            # Critical failure
            if not result.is_valid:
                self.rejection_count += 1
                return False, "I apologize, but I cannot provide that response.", metadata

            # Low confidence - add disclaimer
            if result.confidence < 0.8:
                self.sanitization_count += 1
                safe_output = output + "\n\n(Please verify this information independently.)"
                return True, safe_output, metadata

            return True, output, metadata

        def get_stats(self) -> Dict:
            """Get validation statistics."""
            return {
                "rejections": self.rejection_count,
                "sanitizations": self.sanitization_count,
            }

    # Use validator
    validator = OutputValidator()

    test_outputs = [
        "The capital of France is Paris.",
        "Too short",
        "I don't actually know, but I think maybe...",
    ]

    for output in test_outputs:
        is_valid, safe_output, metadata = validator.validate(output)
        print(f"Input: {output}")
        print(f"  Valid: {is_valid}")
        print(f"  Output: {safe_output[:60]}...")
        print(f"  Confidence: {metadata['confidence']:.2f}")
        print()

    print("Validator stats:", validator.get_stats())


if __name__ == "__main__":
    basic_output_validation()
    json_output_validation()
    confidence_scoring()
    sanitization_strategies()
    validation_pipeline()
    best_practices()
    practical_implementation()

    print("\n" + "=" * 70)
    print("\nKey insight: Output validation is your last line of defense")
    print("Validate everything before it reaches users!")

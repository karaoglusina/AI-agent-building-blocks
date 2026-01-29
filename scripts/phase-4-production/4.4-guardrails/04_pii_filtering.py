"""
04 - PII Filtering
==================
Detect and redact personal information.

Key concept: PII (Personally Identifiable Information) must be detected and redacted before logging, storage, or processing - protecting user privacy and complying with regulations.

Book reference: AI_eng.10.2
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    type: str
    value: str
    start: int
    end: int
    confidence: float


def detect_email(text: str) -> List[PIIMatch]:
    """Detect email addresses."""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = []

    for match in re.finditer(pattern, text):
        matches.append(PIIMatch(
            type="EMAIL",
            value=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.95
        ))

    return matches


def detect_phone(text: str) -> List[PIIMatch]:
    """Detect phone numbers (US format)."""
    patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b',  # 1234567890
    ]

    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            matches.append(PIIMatch(
                type="PHONE",
                value=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.90
            ))

    return matches


def detect_ssn(text: str) -> List[PIIMatch]:
    """Detect Social Security Numbers."""
    pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    matches = []

    for match in re.finditer(pattern, text):
        matches.append(PIIMatch(
            type="SSN",
            value=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.85
        ))

    return matches


def detect_credit_card(text: str) -> List[PIIMatch]:
    """Detect credit card numbers (basic)."""
    # Simplified Luhn algorithm check
    def luhn_check(card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = [int(d) for d in card_number if d.isdigit()]
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    # Match 13-16 digit numbers with optional spaces/dashes
    pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    matches = []

    for match in re.finditer(pattern, text):
        card_number = match.group().replace(' ', '').replace('-', '')
        if luhn_check(card_number):
            matches.append(PIIMatch(
                type="CREDIT_CARD",
                value=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.92
            ))

    return matches


def detect_address(text: str) -> List[PIIMatch]:
    """Detect street addresses (simple heuristic)."""
    # Look for patterns like "123 Main St" or "456 Oak Avenue"
    pattern = r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'
    matches = []

    for match in re.finditer(pattern, text):
        matches.append(PIIMatch(
            type="ADDRESS",
            value=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.70  # Lower confidence - many false positives
        ))

    return matches


def detect_all_pii(text: str) -> List[PIIMatch]:
    """Detect all types of PII in text."""
    all_matches = []

    all_matches.extend(detect_email(text))
    all_matches.extend(detect_phone(text))
    all_matches.extend(detect_ssn(text))
    all_matches.extend(detect_credit_card(text))
    all_matches.extend(detect_address(text))

    # Sort by position
    all_matches.sort(key=lambda x: x.start)

    return all_matches


def redact_pii(text: str, matches: List[PIIMatch], redaction_style: str = "type") -> str:
    """Redact PII from text."""

    if not matches:
        return text

    # Build redacted string
    result = []
    last_end = 0

    for match in matches:
        # Add text before PII
        result.append(text[last_end:match.start])

        # Add redaction
        if redaction_style == "type":
            result.append(f"[{match.type}]")
        elif redaction_style == "asterisk":
            result.append("*" * len(match.value))
        elif redaction_style == "partial":
            if match.type == "EMAIL":
                # Show first char + domain
                parts = match.value.split('@')
                result.append(f"{parts[0][0]}***@{parts[1]}")
            elif match.type == "PHONE":
                # Show last 4 digits
                result.append(f"***-***-{match.value[-4:]}")
            elif match.type == "CREDIT_CARD":
                # Show last 4 digits
                result.append(f"****-****-****-{match.value[-4:]}")
            else:
                result.append(f"[{match.type}]")
        else:
            result.append("[REDACTED]")

        last_end = match.end

    # Add remaining text
    result.append(text[last_end:])

    return ''.join(result)


def basic_pii_detection():
    """Test basic PII detection."""
    print("=== PII DETECTION ===\n")

    test_texts = [
        "My email is john.doe@example.com",
        "Call me at 555-123-4567",
        "My SSN is 123-45-6789",
        "Card number: 4532-1488-0343-6467",
        "I live at 123 Main Street",
        "Contact: john@test.com or (555) 123-4567"]

    for text in test_texts:
        print(f"Text: {text}")
        matches = detect_all_pii(text)

        if matches:
            print(f"  Found {len(matches)} PII item(s):")
            for match in matches:
                print(f"    - {match.type}: {match.value} (confidence: {match.confidence:.2f})")
        else:
            print("  No PII detected")
        print()


def redaction_styles():
    """Show different redaction styles."""
    print("=" * 70)
    print("=== REDACTION STYLES ===\n")

    text = "Email me at john.doe@example.com or call 555-123-4567. Card: 4532-1488-0343-6467"
    matches = detect_all_pii(text)

    print(f"Original: {text}\n")

    styles = ["type", "asterisk", "partial", "full"]

    for style in styles:
        redacted = redact_pii(text, matches, style)
        print(f"{style.capitalize()}: {redacted}")

    print("\n" + "=" * 70)
    print("\nStyle recommendations:")
    print("- Type: Best for logging/debugging (shows what was redacted)")
    print("- Asterisk: Visual indication while preserving length")
    print("- Partial: Balance between privacy and usability")
    print("- Full: Maximum privacy (generic [REDACTED])")


def context_aware_detection():
    """Show context-aware PII detection."""
    print("\n" + "=" * 70)
    print("=== CONTEXT-AWARE DETECTION ===\n")

    def is_pii_in_context(text: str, match: PIIMatch) -> bool:
        """Check if PII appears in a sensitive context."""

        # Get surrounding context (50 chars before/after)
        start = max(0, match.start - 50)
        end = min(len(text), match.end + 50)
        context = text[start:end].lower()

        # Sensitive context indicators
        sensitive_indicators = [
            'password', 'pin', 'ssn', 'social security',
            'credit card', 'account number', 'private',
            'confidential', 'secret', 'login'
        ]

        return any(indicator in context for indicator in sensitive_indicators)

    test_cases = [
        "My email is john@example.com for newsletter",  # Not sensitive
        "Please send the password reset to john@example.com",  # Sensitive
        "Public contact: 555-0100",  # Not sensitive
        "My private number is 555-0100",  # Sensitive
    ]

    for text in test_cases:
        print(f"Text: {text}")
        matches = detect_all_pii(text)

        for match in matches:
            is_sensitive = is_pii_in_context(text, match)
            sensitivity = "SENSITIVE" if is_sensitive else "public"
            print(f"  {match.type}: {match.value} → {sensitivity}")
        print()


def pii_sanitization_pipeline():
    """Show complete PII sanitization pipeline."""
    print("=" * 70)
    print("=== PII SANITIZATION PIPELINE ===\n")

    def sanitize_for_logging(text: str) -> Tuple[str, Dict]:
        """Sanitize text for logging, return redacted text + metadata."""

        # Detect PII
        matches = detect_all_pii(text)

        # Redact
        redacted_text = redact_pii(text, matches, "type")

        # Metadata
        metadata = {
            "original_length": len(text),
            "redacted_length": len(redacted_text),
            "pii_count": len(matches),
            "pii_types": list(set(m.type for m in matches)),
            "has_sensitive_data": len(matches) > 0
        }

        return redacted_text, metadata

    def sanitize_for_llm(text: str) -> str:
        """Sanitize text before sending to LLM."""

        matches = detect_all_pii(text)

        # High-confidence matches only
        high_conf_matches = [m for m in matches if m.confidence > 0.85]

        # Redact with partial visibility for LLM context
        return redact_pii(text, high_conf_matches, "partial")

    # Test pipeline
    user_input = "My email is john.doe@example.com and phone is 555-123-4567. I live at 123 Main Street."

    print("Original input:")
    print(f"  {user_input}\n")

    # For logging
    log_text, metadata = sanitize_for_logging(user_input)
    print("For logging:")
    print(f"  {log_text}")
    print(f"  Metadata: {metadata}\n")

    # For LLM
    llm_text = sanitize_for_llm(user_input)
    print("For LLM processing:")
    print(f"  {llm_text}")


def llm_based_pii_detection():
    """Use LLM for advanced PII detection (example)."""
    print("\n" + "=" * 70)
    print("=== LLM-BASED PII DETECTION ===\n")

    print("Pattern-based detection limitations:")
    print("- Names (many false positives)")
    print("- International formats (non-US phone, address)")
    print("- Contextual PII (job title + company = identifiable)")
    print("- Implicit PII (unique combinations)\n")

    print("LLM-based detection advantages:")
    print("- Understands context better")
    print("- Detects names with higher accuracy")
    print("- Handles international formats")
    print("- Identifies implicit PII\n")

    print("Example prompt for LLM-based detection:")
    prompt = """Identify all personally identifiable information (PII) in this text:

Text: "John Smith works as CTO at AcmeCorp in Seattle"

Return JSON with each PII found:
[
  {"type": "NAME", "value": "John Smith", "start": 0, "end": 10},
  {"type": "JOB_TITLE", "value": "CTO", "start": 21, "end": 24},
  {"type": "ORGANIZATION", "value": "AcmeCorp", "start": 28, "end": 36},
  {"type": "LOCATION", "value": "Seattle", "start": 40, "end": 47}
]"""

    print(prompt)


def best_practices():
    """PII filtering best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Detect Early: Find PII before logging or processing",
        "2. Multiple Detectors: Combine regex + LLM for best coverage",
        "3. Context Matters: Not all emails/phones need redaction",
        "4. Confidence Thresholds: Different actions for different confidence",
        "5. Partial Redaction: Balance privacy with usability",
        "6. Audit Trail: Log what was redacted (without storing PII)",
        "7. User Consent: Allow users to opt-in to data sharing",
        "8. Retention Policies: Delete PII after defined period",
        "9. Encryption: Encrypt PII if storage is necessary",
        "10. Compliance: Follow GDPR, CCPA, HIPAA requirements"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nPII protection layers:")
    print("  Input → Detection → Redaction → Logging")
    print("       → Encryption (if stored)")
    print("       → Audit trail")
    print("       → Retention policy")


def practical_implementation():
    """Show practical PII filtering implementation."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL IMPLEMENTATION ===\n")

    def process_user_message(message: str) -> Dict:
        """Process message with PII protection."""

        # Detect PII
        pii_matches = detect_all_pii(message)

        # Redact for logging
        log_safe = redact_pii(message, pii_matches, "type")

        # Redact for LLM (partial, high confidence only)
        high_conf = [m for m in pii_matches if m.confidence > 0.85]
        llm_safe = redact_pii(message, high_conf, "partial")

        return {
            "original_length": len(message),
            "log_safe_message": log_safe,
            "llm_safe_message": llm_safe,
            "pii_detected": len(pii_matches) > 0,
            "pii_types": [m.type for m in pii_matches],
            "warning": "Message contains PII" if pii_matches else None
        }

    # Test
    message = "Contact me at john@test.com or 555-0100"
    result = process_user_message(message)

    print("Input:", message)
    print("\nProcessing result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    basic_pii_detection()
    redaction_styles()
    context_aware_detection()
    pii_sanitization_pipeline()
    llm_based_pii_detection()
    best_practices()
    practical_implementation()

    print("\n" + "=" * 70)
    print("\nKey insight: PII protection is multi-layered")
    print("Detect → Redact → Log → Monitor → Delete")

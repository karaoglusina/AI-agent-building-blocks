"""
01 - Input Validation
=====================
Validate user input before processing.

Key concept: Input validation prevents malformed requests from reaching your LLM - reducing costs, improving security, and providing better user feedback.

Book reference: AI_eng.10.2
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Optional, List
import re


class ChatRequest(BaseModel):
    """Validate chat input with Pydantic."""
    message: str = Field(..., min_length=1, max_length=5000)
    user_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    max_tokens: Optional[int] = Field(default=500, ge=1, le=4000)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

    @validator('message')
    def message_not_empty(cls, v):
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

    @validator('message')
    def check_excessive_repetition(cls, v):
        """Detect spam patterns (repeated characters)."""
        # Check for excessive repeated characters (e.g., "aaaaaaa...")
        if re.search(r'(.)\1{50,}', v):
            raise ValueError('Message contains excessive character repetition')
        return v


class SearchRequest(BaseModel):
    """Validate search query."""
    query: str = Field(..., min_length=2, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    filters: Optional[List[str]] = Field(default=None, max_items=10)

    @validator('query')
    def sanitize_query(cls, v):
        """Remove potentially harmful characters."""
        # Remove control characters
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return v.strip()


def validate_input_basic():
    """Basic input validation examples."""
    print("=== BASIC INPUT VALIDATION ===\n")

    # Valid input
    try:
        request = ChatRequest(
            message="What's the weather like today?",
            user_id="user_123",
            max_tokens=100
        )
        print("✓ Valid request:")
        print(f"  Message: {request.message}")
        print(f"  User: {request.user_id}")
        print(f"  Max tokens: {request.max_tokens}\n")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}\n")

    # Invalid: empty message
    print("--- Testing empty message ---")
    try:
        request = ChatRequest(
            message="   ",
            user_id="user_123"
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")

    # Invalid: message too long
    print("--- Testing message too long ---")
    try:
        request = ChatRequest(
            message="x" * 6000,
            user_id="user_123"
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")

    # Invalid: bad user_id format
    print("--- Testing invalid user_id ---")
    try:
        request = ChatRequest(
            message="Hello",
            user_id="user@123"  # @ not allowed
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")


def validate_numeric_constraints():
    """Test numeric field validation."""
    print("=" * 70)
    print("=== NUMERIC CONSTRAINTS ===\n")

    # Invalid temperature
    print("--- Testing temperature out of range ---")
    try:
        request = ChatRequest(
            message="Hello",
            user_id="user_123",
            temperature=3.0  # Too high
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")

    # Invalid max_tokens
    print("--- Testing max_tokens out of range ---")
    try:
        request = ChatRequest(
            message="Hello",
            user_id="user_123",
            max_tokens=5000  # Too high
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")


def validate_spam_patterns():
    """Detect spam and abuse patterns."""
    print("=" * 70)
    print("=== SPAM PATTERN DETECTION ===\n")

    # Excessive repetition
    print("--- Testing excessive character repetition ---")
    try:
        request = ChatRequest(
            message="a" * 100,
            user_id="user_123"
        )
        print("✓ Request accepted\n")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e.errors()[0]['msg']}\n")


def custom_validation_rules():
    """Show custom validation patterns."""
    print("=" * 70)
    print("=== CUSTOM VALIDATION RULES ===\n")

    def validate_business_rules(request: ChatRequest) -> tuple[bool, Optional[str]]:
        """Apply domain-specific validation rules."""

        # Rule 1: Restrict certain words in production
        forbidden_words = ['admin', 'root', 'system']
        message_lower = request.message.lower()
        for word in forbidden_words:
            if word in message_lower:
                return False, f"Message contains restricted word: '{word}'"

        # Rule 2: Check for SQL injection patterns (basic)
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'insert\s+into',
            r'delete\s+from'
        ]
        for pattern in sql_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return False, f"Message contains suspicious pattern"

        # Rule 3: Limit consecutive questions (rate limiting indicator)
        # This would be checked against a database in production

        return True, None

    # Test valid message
    print("--- Testing valid message ---")
    request = ChatRequest(message="What's the capital of France?", user_id="user_123")
    valid, error = validate_business_rules(request)
    if valid:
        print("✓ Validation passed\n")
    else:
        print(f"✗ Validation failed: {error}\n")

    # Test restricted word
    print("--- Testing restricted word ---")
    request = ChatRequest(message="Show me admin panel", user_id="user_123")
    valid, error = validate_business_rules(request)
    if valid:
        print("✓ Validation passed\n")
    else:
        print(f"✗ Validation failed: {error}\n")

    # Test SQL pattern
    print("--- Testing SQL injection pattern ---")
    request = ChatRequest(message="' UNION SELECT * FROM users --", user_id="user_123")
    valid, error = validate_business_rules(request)
    if valid:
        print("✓ Validation passed\n")
    else:
        print(f"✗ Validation failed: {error}\n")


def validation_best_practices():
    """Show validation best practices."""
    print("=" * 70)
    print("=== VALIDATION BEST PRACTICES ===\n")

    practices = [
        "1. Validate Early: Check input before expensive LLM calls",
        "2. Clear Errors: Return specific, actionable error messages",
        "3. Fail Fast: Reject invalid input immediately",
        "4. Whitelist > Blacklist: Allow known good patterns",
        "5. Sanitize: Clean input even if it passes validation",
        "6. Log Failures: Track validation failures for security monitoring",
        "7. Rate Limiting: Track per-user request counts",
        "8. Cost Protection: Enforce token limits per request",
        "9. Type Safety: Use Pydantic for automatic type validation",
        "10. Domain Rules: Apply business-specific validation logic"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nValidation flow:")
    print("  User Input → Type Validation → Format Validation")
    print("              → Business Rules → Sanitization → LLM")


if __name__ == "__main__":
    validate_input_basic()
    validate_numeric_constraints()
    validate_spam_patterns()
    custom_validation_rules()
    validation_best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Input validation is your first line of defense")
    print("Catch bad input early - before it costs you money or causes harm!")

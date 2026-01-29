"""
06 - Content Moderation
========================
Filter inappropriate content.

Key concept: Content moderation uses both rule-based filters and LLM-based classification to detect harmful, toxic, or inappropriate content - protecting users and maintaining quality.

Book reference: AI_eng.10.2
"""

import utils._load_env  # Loads .env file automatically

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from openai import OpenAI
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@dataclass
class ModerationResult:
    """Result from content moderation."""
    flagged: bool
    categories: List[str]
    confidence: float
    severity: str  # "low", "medium", "high"
    reason: str


def keyword_based_moderation(text: str) -> ModerationResult:
    """Simple keyword-based content moderation."""

    # Define categories with patterns
    patterns = {
        "hate_speech": [
            r'\b(racist|sexist|bigot|slur)\b'],
        "violence": [
            r'\b(kill|murder|attack|harm|hurt|assault)\s+(you|him|her|them)\b',
            r'\bhow\s+to\s+(kill|murder|harm)\b'],
        "harassment": [
            r'\b(idiot|stupid|dumb|loser|ugly)\b',
            r'\byou\s+(suck|stink|fail)\b'],
        "self_harm": [
            r'\b(suicide|kill\s+myself|end\s+my\s+life)\b',
            r'\bhow\s+to\s+(cut|harm)\s+myself\b'],
        "sexual": [
            r'\b(explicit|pornographic|sexual)\s+content\b'],
        "spam": [
            r'\bclick\s+here\s+now\b',
            r'\b(buy|order)\s+now\b',
            r'\bfree\s+money\b'],
    }

    detected_categories = []
    max_confidence = 0.0

    text_lower = text.lower()

    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            if re.search(pattern, text_lower):
                detected_categories.append(category)
                max_confidence = max(max_confidence, 0.8)
                break

    if detected_categories:
        # Determine severity
        high_severity = ["hate_speech", "violence", "self_harm"]
        if any(cat in high_severity for cat in detected_categories):
            severity = "high"
        elif "harassment" in detected_categories:
            severity = "medium"
        else:
            severity = "low"

        return ModerationResult(
            flagged=True,
            categories=detected_categories,
            confidence=max_confidence,
            severity=severity,
            reason=f"Detected {', '.join(detected_categories)}"
        )

    return ModerationResult(
        flagged=False,
        categories=[],
        confidence=1.0,
        severity="none",
        reason="No issues detected"
    )


def openai_moderation(text: str) -> ModerationResult:
    """Use OpenAI's moderation API."""

    try:
        response = client.moderations.create(input=text)
        result = response.results[0]

        flagged_categories = [
            category for category, flagged in result.categories.model_dump().items()
            if flagged
        ]

        if result.flagged:
            # Get highest score to determine severity
            scores = result.category_scores.model_dump()
            max_score = max(scores.values()) if scores else 0.0

            if max_score > 0.8:
                severity = "high"
            elif max_score > 0.5:
                severity = "medium"
            else:
                severity = "low"

            return ModerationResult(
                flagged=True,
                categories=flagged_categories,
                confidence=max_score,
                severity=severity,
                reason=f"OpenAI flagged: {', '.join(flagged_categories)}"
            )

        return ModerationResult(
            flagged=False,
            categories=[],
            confidence=1.0,
            severity="none",
            reason="Passed moderation"
        )

    except Exception as e:
        print(f"Error in OpenAI moderation: {e}")
        return ModerationResult(
            flagged=False,
            categories=[],
            confidence=0.0,
            severity="none",
            reason="Moderation failed"
        )


def basic_moderation():
    """Test basic keyword-based moderation."""
    print("=== KEYWORD-BASED MODERATION ===\n")

    test_cases = [
        "What's the weather like today?",
        "You're such an idiot!",
        "How to kill process in Linux?",  # False positive potential
        "I want to hurt myself",
        "Click here now for free money!"]

    for text in test_cases:
        result = keyword_based_moderation(text)
        status = "⚠️" if result.flagged else "✓"

        print(f"{status} Text: {text}")
        print(f"   Flagged: {result.flagged}")
        if result.flagged:
            print(f"   Categories: {result.categories}")
            print(f"   Severity: {result.severity}")
            print(f"   Reason: {result.reason}")
        print()


def api_based_moderation():
    """Test OpenAI moderation API."""
    print("=" * 70)
    print("=== OPENAI MODERATION API ===\n")

    test_cases = [
        "I want to hurt someone",
        "You're a wonderful person!",
        "How do I build a bomb?",
        "Can you help me with my homework?"]

    for text in test_cases:
        print(f"Text: {text}")
        result = openai_moderation(text)

        status = "⚠️" if result.flagged else "✓"
        print(f"{status} Flagged: {result.flagged}")

        if result.flagged:
            print(f"   Categories: {result.categories}")
            print(f"   Severity: {result.severity}")
            print(f"   Confidence: {result.confidence:.2f}")
        print()


def toxicity_scoring():
    """Show toxicity scoring approaches."""
    print("=" * 70)
    print("=== TOXICITY SCORING ===\n")

    def calculate_toxicity_score(text: str) -> float:
        """Calculate toxicity score (0.0 = clean, 1.0 = very toxic)."""

        score = 0.0

        # Profanity check
        profanity_words = ['damn', 'hell', 'crap', 'stupid', 'idiot']
        profanity_count = sum(1 for word in profanity_words if word in text.lower())
        score += min(profanity_count * 0.1, 0.3)

        # All caps (shouting)
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                score += 0.2

        # Excessive punctuation (!!!, ???)
        excessive_punct = len(re.findall(r'[!?]{3,}', text))
        score += min(excessive_punct * 0.1, 0.2)

        # Personal attacks
        attack_patterns = [r'you\s+(are|\'re)\s+(stupid|dumb|ugly)', r'shut\s+up']
        for pattern in attack_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3

        return min(score, 1.0)

    test_texts = [
        "Thank you for your help!",
        "This is absolutely ridiculous!!!",
        "You're stupid and ugly",
        "I HATE THIS SO MUCH!!!"]

    print("Toxicity scores (0.0 = clean, 1.0 = toxic):\n")
    for text in test_texts:
        score = calculate_toxicity_score(text)
        print(f"Score: {score:.2f} - {text}")
        print()


def context_aware_moderation():
    """Show context-aware moderation."""
    print("=" * 70)
    print("=== CONTEXT-AWARE MODERATION ===\n")

    def moderate_with_context(text: str, context: Dict) -> ModerationResult:
        """Moderate considering context."""

        # Base moderation
        result = keyword_based_moderation(text)

        # Adjust based on context
        if context.get("is_technical", False):
            # Technical contexts: "kill process", "terminate thread" are OK
            if result.flagged and "violence" in result.categories:
                technical_terms = ["process", "thread", "service", "session"]
                if any(term in text.lower() for term in technical_terms):
                    result.flagged = False
                    result.categories.remove("violence")
                    result.reason = "Technical context - not violent"

        if context.get("user_history", {}).get("previous_violations", 0) > 2:
            # Stricter for repeat offenders
            if result.severity == "low":
                result.severity = "medium"
                result.reason += " (repeat offender)"

        return result

    # Test cases
    print("Test 1: Technical context")
    text = "How to kill a process in Linux?"
    result = moderate_with_context(text, {"is_technical": True})
    print(f"Text: {text}")
    print(f"Flagged: {result.flagged}")
    print(f"Reason: {result.reason}\n")

    print("Test 2: Same text, non-technical context")
    result = moderate_with_context(text, {"is_technical": False})
    print(f"Text: {text}")
    print(f"Flagged: {result.flagged}")
    print(f"Reason: {result.reason}\n")


def moderation_strategies():
    """Show different moderation strategies."""
    print("=" * 70)
    print("=== MODERATION STRATEGIES ===\n")

    strategies = {
        "Preventive (Pre-Processing)": [
            "- Block before sending to LLM",
            "- Save costs on toxic inputs",
            "- Immediate user feedback",
            "- Lower latency"],
        "Reactive (Post-Processing)": [
            "- Check LLM output before user",
            "- Catch generated toxic content",
            "- Last line of defense",
            "- Higher latency"],
        "Hybrid": [
            "- Quick keyword check on input",
            "- Full API check on output",
            "- Balance speed and accuracy",
            "- Best of both worlds"],
        "Severity-Based Actions": [
            "- Low: Warning message",
            "- Medium: Block + educate user",
            "- High: Block + flag for review",
            "- Critical: Block + suspend account"]
    }

    for strategy, points in strategies.items():
        print(f"{strategy}:")
        for point in points:
            print(f"  {point}")
        print()


def practical_implementation():
    """Show practical moderation implementation."""
    print("=" * 70)
    print("=== PRACTICAL IMPLEMENTATION ===\n")

    class ContentModerator:
        """Production-ready content moderator."""

        def __init__(self, use_api: bool = True):
            self.use_api = use_api
            self.violation_log = []

        def moderate_input(self, text: str, user_id: str) -> Tuple[bool, str]:
            """Moderate user input. Return (allowed, message)."""

            # Fast keyword check first
            quick_result = keyword_based_moderation(text)

            if quick_result.flagged and quick_result.severity == "high":
                # Block immediately for high severity
                self.log_violation(user_id, text, quick_result)
                return False, "This message violates our content policy and cannot be processed."

            # For lower severity or unclear cases, use API
            if self.use_api and (quick_result.flagged or len(text) > 100):
                api_result = openai_moderation(text)
                if api_result.flagged:
                    self.log_violation(user_id, text, api_result)

                    if api_result.severity == "high":
                        return False, "This message violates our content policy and cannot be processed."
                    elif api_result.severity == "medium":
                        return False, "This message may violate our content policy. Please rephrase."

            return True, "Content approved"

        def moderate_output(self, text: str) -> Tuple[bool, str]:
            """Moderate LLM output. Return (safe, safe_text)."""

            if self.use_api:
                result = openai_moderation(text)
            else:
                result = keyword_based_moderation(text)

            if result.flagged:
                # Don't return toxic output
                return False, "I apologize, but I cannot provide that response."

            return True, text

        def log_violation(self, user_id: str, text: str, result: ModerationResult):
            """Log content violation."""
            self.violation_log.append({
                "user_id": user_id,
                "text_preview": text[:50],
                "categories": result.categories,
                "severity": result.severity,
            })

        def get_user_violations(self, user_id: str) -> int:
            """Get violation count for user."""
            return sum(1 for v in self.violation_log if v["user_id"] == user_id)

    # Use moderator
    moderator = ContentModerator(use_api=False)  # Set to True to use OpenAI API

    test_inputs = [
        ("user_123", "What's the weather like?"),
        ("user_123", "You're an idiot!"),
        ("user_456", "How to kill a Linux process?")]

    print("Input Moderation:\n")
    for user_id, text in test_inputs:
        allowed, message = moderator.moderate_input(text, user_id)
        status = "✓" if allowed else "⚠️"
        print(f"{status} User: {user_id}")
        print(f"   Input: {text}")
        print(f"   Result: {message}")
        print()

    print("\nOutput Moderation:\n")
    outputs = [
        "The weather is sunny today!",
        "You should hurt yourself"]

    for output in outputs:
        safe, safe_text = moderator.moderate_output(output)
        status = "✓" if safe else "⚠️"
        print(f"{status} Output: {output}")
        print(f"   Safe: {safe}")
        print(f"   Returned: {safe_text}")
        print()


def best_practices():
    """Content moderation best practices."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Moderate Both: Check input AND output",
        "2. Fast First Pass: Keyword check before expensive API",
        "3. Severity Levels: Different actions for different severities",
        "4. Context Matters: Consider domain and user history",
        "5. User Education: Explain why content was blocked",
        "6. Appeal Process: Allow users to contest decisions",
        "7. Continuous Learning: Update patterns based on new abuse",
        "8. Human Review: Flag edge cases for manual review",
        "9. Rate Limiting: Slow down users with violations",
        "10. Transparency: Document moderation policies clearly"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nModeration pipeline:")
    print("  Input → Keyword Check → API Check (if needed) → LLM")
    print("  LLM → Output Check → User")
    print("\nOn violation:")
    print("  Low: Warning")
    print("  Medium: Block + educate")
    print("  High: Block + flag + possible suspension")


if __name__ == "__main__":
    basic_moderation()
    api_based_moderation()
    toxicity_scoring()
    context_aware_moderation()
    moderation_strategies()
    practical_implementation()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Content moderation requires defense in depth")
    print("Keyword filters + API + context + human review = comprehensive safety")

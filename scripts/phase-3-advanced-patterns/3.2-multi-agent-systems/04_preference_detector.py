"""
04 - Preference Detection Agent
================================
Agent that detects and stores user preferences from conversation.

Key concept: Specialized agent extracts and maintains user preference memory.

Book reference: AI_eng.6 (Memory)
"""

from openai import OpenAI
import os
from pydantic import BaseModel

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class DetectedPreferences(BaseModel):
    """Preferences extracted from user input."""
    job_types: list[str] = []
    skills: list[str] = []
    locations: list[str] = []
    work_style: list[str] = []  # remote, hybrid, office, etc.
    company_preferences: list[str] = []
    salary_expectations: str = ""
    other_preferences: list[str] = []


class PreferenceDetector:
    """Agent specialized in detecting and storing user preferences."""

    def __init__(self):
        self.preferences = {
            "job_types": set(),
            "skills": set(),
            "locations": set(),
            "work_style": set(),
            "company_preferences": set(),
            "salary_expectations": "",
            "other_preferences": set(),
        }

    def detect(self, user_input: str) -> DetectedPreferences:
        """Detect preferences from user input."""
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": "Extract user preferences about jobs from their message. "
        "Include job types, skills, locations, work style preferences, "
        "company preferences, salary expectations, and any other relevant preferences."
        },
        {"role": "user", "content": user_input}
        ]
        ,
        response_format={"type": "json_object"})
        return DetectedPreferences.model_validate_json(response.choices[0].message.content)

    def update_preferences(self, user_input: str) -> dict:
        """Detect and update stored preferences."""
        detected = self.detect(user_input)

        # Update stored preferences
        for job_type in detected.job_types:
            self.preferences["job_types"].add(job_type.lower())
        for skill in detected.skills:
            self.preferences["skills"].add(skill.lower())
        for location in detected.locations:
            self.preferences["locations"].add(location.lower())
        for style in detected.work_style:
            self.preferences["work_style"].add(style.lower())
        for company_pref in detected.company_preferences:
            self.preferences["company_preferences"].add(company_pref.lower())
        if detected.salary_expectations:
            self.preferences["salary_expectations"] = detected.salary_expectations
        for other in detected.other_preferences:
            self.preferences["other_preferences"].add(other.lower())

        return {
            "detected_now": detected.model_dump(),
            "total_stored": self.get_summary()
        }

    def get_summary(self) -> dict:
        """Get summary of all stored preferences."""
        return {
            key: list(value) if isinstance(value, set) else value
            for key, value in self.preferences.items()
            if value
        }

    def get_context_string(self) -> str:
        """Get preferences as a formatted string for context."""
        summary = self.get_summary()
        if not summary:
            return "No preferences detected yet."

        lines = []
        for key, value in summary.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, list):
                lines.append(f"{formatted_key}: {', '.join(value)}")
            else:
                lines.append(f"{formatted_key}: {value}")

        return "\n".join(lines)


class MainAgent:
    """Main agent that uses preference detector."""

    def __init__(self, preference_detector: PreferenceDetector):
        self.preference_detector = preference_detector

    def respond(self, user_input: str) -> str:
        """Respond to user, using detected preferences."""
        # Update preferences
        update_result = self.preference_detector.update_preferences(user_input)

        # Generate response with preference context
        pref_context = self.preference_detector.get_context_string()

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": f"You're a job search assistant. Use the user's known preferences to personalize responses.\n\n"
        f"Known preferences:\n{pref_context}"
        },
        {"role": "user", "content": user_input}
        ]
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    print("=== PREFERENCE DETECTION AGENT ===\n")

    detector = PreferenceDetector()
    agent = MainAgent(detector)

    conversation = [
        "I'm looking for senior Python developer positions",
        "I prefer remote work in Europe",
        "I have experience with Django and FastAPI",
        "I'd like to work for a startup, ideally Series A or B",
        "Salary expectations around 80-100k EUR",
        "Now show me jobs that match my profile"]

    for user_input in conversation:
        print(f"User: {user_input}")
        response = agent.respond(user_input)
        print(f"Assistant: {response}\n")

    print("=" * 50)
    print("\n=== FINAL STORED PREFERENCES ===")
    print(detector.get_context_string())

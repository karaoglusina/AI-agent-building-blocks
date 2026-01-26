"""
01 - Structured Memory
======================
Store memories in typed schemas.

Key concept: Use Pydantic models to enforce memory structure and validation.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from openai import OpenAI
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

client = OpenAI()


class UserPreference(BaseModel):
    """Structured preference memory."""
    preference_type: Literal["work_style", "location", "company_size", "industry"]
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    mentioned_at: datetime = Field(default_factory=datetime.now)


class UserSkill(BaseModel):
    """Structured skill memory."""
    skill_name: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    years_experience: int | None = None
    mentioned_at: datetime = Field(default_factory=datetime.now)


class UserGoal(BaseModel):
    """Structured goal memory."""
    goal: str
    priority: Literal["low", "medium", "high"]
    target_date: str | None = None
    mentioned_at: datetime = Field(default_factory=datetime.now)


class StructuredMemoryExtraction(BaseModel):
    """Extraction result containing structured memories."""
    preferences: list[UserPreference] = []
    skills: list[UserSkill] = []
    goals: list[UserGoal] = []


class StructuredMemoryStore:
    """Memory store with typed schemas."""

    def __init__(self):
        self.preferences: list[UserPreference] = []
        self.skills: list[UserSkill] = []
        self.goals: list[UserGoal] = []

    def extract_structured_memories(self, conversation: str) -> StructuredMemoryExtraction:
        """Extract structured memories from conversation."""
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": "Extract structured information about the user's preferences, "
                               "skills, and goals. Be specific and assign appropriate confidence levels."
                },
                {"role": "user", "content": conversation}
            ],
            text_format=StructuredMemoryExtraction
        )
        return response.output_parsed

    def add_memories(self, extraction: StructuredMemoryExtraction):
        """Add extracted memories to store."""
        self.preferences.extend(extraction.preferences)
        self.skills.extend(extraction.skills)
        self.goals.extend(extraction.goals)

    def get_summary(self) -> str:
        """Get human-readable summary of all memories."""
        lines = ["=== STRUCTURED MEMORY STORE ===\n"]

        if self.preferences:
            lines.append("PREFERENCES:")
            for pref in self.preferences:
                lines.append(f"  • {pref.preference_type}: {pref.value} (confidence: {pref.confidence:.0%})")

        if self.skills:
            lines.append("\nSKILLS:")
            for skill in self.skills:
                exp = f", {skill.years_experience}y exp" if skill.years_experience else ""
                lines.append(f"  • {skill.skill_name}: {skill.proficiency}{exp}")

        if self.goals:
            lines.append("\nGOALS:")
            for goal in self.goals:
                target = f" by {goal.target_date}" if goal.target_date else ""
                lines.append(f"  • [{goal.priority}] {goal.goal}{target}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("=== STRUCTURED MEMORY ===\n")

    store = StructuredMemoryStore()

    # Example conversation
    conversation = """
    User: I'm a senior Python developer with 7 years of experience.
    User: I also know JavaScript at an intermediate level.
    User: I strongly prefer remote work and want to work for a startup.
    User: My goal is to become a tech lead within the next 2 years.
    User: I'm interested in the fintech industry.
    """

    print("Extracting structured memories from conversation...")
    extraction = store.extract_structured_memories(conversation)

    print(f"\nExtracted:")
    print(f"  {len(extraction.preferences)} preferences")
    print(f"  {len(extraction.skills)} skills")
    print(f"  {len(extraction.goals)} goals")

    # Add to store
    store.add_memories(extraction)

    # Display structured memory
    print(f"\n{store.get_summary()}")

    # Show type validation benefit
    print("\n=== TYPE SAFETY EXAMPLE ===")
    try:
        invalid = UserPreference(
            preference_type="invalid_type",  # Will fail validation
            value="test",
            confidence=0.5
        )
    except Exception as e:
        print(f"✓ Validation caught error: {type(e).__name__}")

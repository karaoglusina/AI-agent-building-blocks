"""
07 - Schema Inspection
======================
Understand the JSON schema that Pydantic generates.

Key concept: OpenAI uses the JSON schema under the hood.
"""

import json
from pydantic import BaseModel, Field
from typing import Literal


class SkillRequirement(BaseModel):
    """A single skill requirement."""
    name: str = Field(description="Name of the skill")
    level: Literal["beginner", "intermediate", "advanced"]
    is_required: bool = True


class JobRequirements(BaseModel):
    """Requirements extracted from a job posting."""
    title: str = Field(description="Job title")
    min_years: int = Field(ge=0, le=30, description="Minimum years of experience")
    skills: list[SkillRequirement] = Field(description="List of required skills")
    education: str | None = Field(default=None, description="Required education level")


# Get the JSON schema
schema = JobRequirements.model_json_schema()

print("JSON Schema (what OpenAI sees):")
print(json.dumps(schema, indent=2))

# Key points:
# - "type" tells OpenAI the expected type
# - "properties" defines the fields
# - "required" lists mandatory fields
# - "description" helps the model understand what to extract
# - "$defs" contains nested model definitions

print("\n" + "=" * 50)
print("This schema tells OpenAI:")
print("1. What fields to output")
print("2. What types each field should be")
print("3. What values are allowed (e.g., Literal)")
print("4. What each field means (via descriptions)")

"""
08 - Enums and Literals
=======================
Restrict field values to specific options.

Key concept: Use Enum or Literal for categorical fields.
"""

from enum import Enum
from typing import Literal
from pydantic import BaseModel


# Method 1: Enum (when you need to reference values elsewhere)
class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class ContractType(str, Enum):
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    CONTRACT = "contract"
    FREELANCE = "freelance"


class JobWithEnum(BaseModel):
    title: str
    experience: ExperienceLevel
    contract: ContractType


# Method 2: Literal (simpler, inline definition)
class JobWithLiteral(BaseModel):
    title: str
    experience: Literal["entry", "mid", "senior", "lead"]
    contract: Literal["full-time", "part-time", "contract", "freelance"]


# Both work the same way
job1 = JobWithEnum(
    title="Data Analyst",
    experience=ExperienceLevel.MID,  # Use enum value
    contract="full-time"  # String also works
)
print(f"Enum: {job1.experience.value}, {job1.contract.value}")

job2 = JobWithLiteral(
    title="Engineer",
    experience="senior",
    contract="contract"
)
print(f"Literal: {job2.experience}, {job2.contract}")

# Invalid value raises error
try:
    bad_job = JobWithLiteral(
        title="Manager",
        experience="expert",  # Not in allowed values!
        contract="full-time"
    )
except Exception as e:
    print(f"\nValidation error: {e}")

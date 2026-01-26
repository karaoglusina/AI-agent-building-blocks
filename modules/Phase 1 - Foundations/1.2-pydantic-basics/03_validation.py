"""
03 - Field Validation
=====================
Add constraints to fields beyond just type checking.

Key concept: Field() adds validation rules, validators add custom logic.
"""

from pydantic import BaseModel, Field, field_validator


class JobPosting(BaseModel):
    # Field constraints
    title: str = Field(min_length=3, max_length=100)
    salary: int = Field(ge=0, le=1000000)  # ge=greater/equal, le=less/equal
    years_required: int = Field(ge=0, le=50, default=0)
    
    # Custom validation with @field_validator
    location: str
    
    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        # Custom validation logic
        if len(v) < 2:
            raise ValueError("Location must be at least 2 characters")
        return v.title()  # Also transform: "amsterdam" → "Amsterdam"


# Valid data
job = JobPosting(
    title="Data Analyst",
    salary=60000,
    location="amsterdam"
)
print(job)
print(f"Location was transformed: {job.location}")
print()

# Invalid: salary too high
try:
    bad_job = JobPosting(title="CEO", salary=9999999, location="NYC")
except Exception as e:
    print(f"Validation error: {e}")

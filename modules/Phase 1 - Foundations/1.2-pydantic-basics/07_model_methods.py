"""
07 - Model Methods
==================
Add custom methods to your models for computed properties.

Key concept: Models are classes - add methods for behavior.
"""

from pydantic import BaseModel, computed_field


class JobPosting(BaseModel):
    title: str
    company: str
    min_salary: int
    max_salary: int
    skills: list[str]
    description: str
    
    # Computed field - calculated from other fields
    @computed_field
    @property
    def salary_range(self) -> str:
        return f"€{self.min_salary:,} - €{self.max_salary:,}"
    
    @computed_field
    @property
    def avg_salary(self) -> int:
        return (self.min_salary + self.max_salary) // 2
    
    # Regular method
    def has_skill(self, skill: str) -> bool:
        """Check if job requires a specific skill (case-insensitive)."""
        return skill.lower() in [s.lower() for s in self.skills]
    
    def summary(self, max_length: int = 100) -> str:
        """Get a truncated summary of the job."""
        desc = self.description[:max_length]
        return f"{desc}..." if len(self.description) > max_length else desc


# Create job
job = JobPosting(
    title="Senior Python Developer",
    company="TechCorp",
    min_salary=70000,
    max_salary=90000,
    skills=["Python", "Django", "PostgreSQL"],
    description="We are looking for an experienced Python developer to join our team and build amazing products."
)

# Use computed fields
print(f"Salary: {job.salary_range}")
print(f"Average: €{job.avg_salary:,}")
print()

# Use methods
print(f"Has Python? {job.has_skill('python')}")
print(f"Has Java? {job.has_skill('Java')}")
print()

print(f"Summary: {job.summary(50)}")

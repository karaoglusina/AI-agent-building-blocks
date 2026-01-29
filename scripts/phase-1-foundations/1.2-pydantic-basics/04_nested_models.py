"""
04 - Nested Models
==================
Models can contain other models for complex data structures.

Key concept: Compose models for hierarchical data.
"""

from pydantic import BaseModel


class Company(BaseModel):
    name: str
    industry: str
    size: str  # "startup", "medium", "enterprise"


class Location(BaseModel):
    city: str
    country: str
    is_remote: bool = False


class JobPosting(BaseModel):
    title: str
    company: Company  # Nested model
    location: Location  # Nested model
    skills: list[str]


# Create with nested data
job = JobPosting(
    title="Senior Data Engineer",
    company=Company(name="TechCorp", industry="Technology", size="enterprise"),
    location=Location(city="Amsterdam", country="Netherlands", is_remote=True),
    skills=["Python", "Spark", "SQL"]
)

print(job)
print()

# Access nested fields
print(f"Company: {job.company.name}")
print(f"City: {job.location.city}")
print(f"Remote: {job.location.is_remote}")
print()

# Can also create from nested dictionaries
job2 = JobPosting(
    title="Product Manager",
    company={"name": "StartupXYZ", "industry": "Fintech", "size": "startup"},
    location={"city": "Rotterdam", "country": "Netherlands"},
    skills=["Agile", "SQL"]
)
print(f"From dict: {job2.company.name} in {job2.location.city}")

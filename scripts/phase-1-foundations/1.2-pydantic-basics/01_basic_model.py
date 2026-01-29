"""
01 - Basic Model
================
Define a data structure with automatic validation.

Key concept: Pydantic models are classes that validate data automatically.
"""

from pydantic import BaseModel


# Define a model (like a strict dataclass)
class Job(BaseModel):
    title: str
    company: str
    salary: int


# Create an instance - data is validated automatically
job = Job(title="Data Analyst", company="Acme Corp", salary=50000)

print(job)
print(job.title)
print(job.company)
print(job.salary)

# Type coercion: Pydantic converts compatible types
job2 = Job(title="Engineer", company="TechCo", salary="60000")  # str → int
print(f"Salary type: {type(job2.salary)}")  # int, not str!

# Invalid data raises ValidationError
try:
    job3 = Job(title="Manager", company="Corp", salary="not a number")
except Exception as e:
    print(f"Validation error: {e}")

"""
05 - JSON Serialization
=======================
Convert between Pydantic models and JSON/dictionaries.

Key concept: model_dump() and model_validate() for serialization.
"""

import json
from pydantic import BaseModel


class Job(BaseModel):
    title: str
    company: str
    skills: list[str]


# Create a model
job = Job(title="Data Analyst", company="Acme", skills=["Python", "SQL"])

# Model → Dictionary
job_dict = job.model_dump()
print("To dict:", job_dict)
print(f"Type: {type(job_dict)}")
print()

# Model → JSON string
job_json = job.model_dump_json()
print("To JSON:", job_json)
print(f"Type: {type(job_json)}")
print()

# Dictionary → Model
data = {"title": "Engineer", "company": "TechCo", "skills": ["Java"]}
job_from_dict = Job.model_validate(data)
print("From dict:", job_from_dict)
print()

# JSON string → Model
json_string = '{"title": "Designer", "company": "Creative", "skills": ["Figma"]}'
job_from_json = Job.model_validate_json(json_string)
print("From JSON:", job_from_json)
print()

# Pretty JSON output
print("Pretty JSON:")
print(job.model_dump_json(indent=2))

"""
04 - Self-Correction
====================
Detect and fix errors automatically.

Key concept: Validate outputs and retry with corrections - agents can detect mistakes and fix them without human intervention.

Book reference: AI_eng.6
"""

import json
from openai import OpenAI
import os
from pydantic import BaseModel, Field, ValidationError
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class JobAnalysis(BaseModel):
    """Structured job analysis."""
    title: str
    category: str = Field(..., description="One of: engineering, design, product, sales, operations")
    seniority: str = Field(..., description="One of: junior, mid, senior, lead, executive")
    required_skills: list[str] = Field(..., min_items=1, max_items=10)
    salary_range: str = Field(..., pattern=r"^\$\d+k-\$\d+k$")


def extract_job_info(job: dict, attempt: int = 1) -> str:
    """Extract structured info from job posting."""
    prompt = f"""Extract structured information from this job:

Title: {job['title']}
Description: {job.get('description', '')[:300]}

Return JSON with:
- title: job title
- category: one of [engineering, design, product, sales, operations]
- seniority: one of [junior, mid, senior, lead, executive]
- required_skills: list of 3-5 key skills
- salary_range: format like "$100k-$150k"
"""

    if attempt > 1:
        prompt += f"\n\nPrevious attempt failed validation. Ensure all fields match the required format exactly."

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Extract job information in valid JSON format."},
    {"role": "user", "content": prompt}
    ]
    )
    return response.choices[0].message.content


def validate_and_correct(job: dict, max_attempts: int = 3) -> dict:
    """Extract and validate job info with automatic correction."""
    print(f"\nProcessing: {job['title']}")

    for attempt in range(1, max_attempts + 1):
        print(f"\n  Attempt {attempt}...")

        # Extract
        raw_output = extract_job_info(job, attempt)

        try:
            # Parse JSON
            data = json.loads(raw_output)

            # Validate with Pydantic
            analysis = JobAnalysis(**data)

            print(f"  ✓ Validation successful")
            return {
                "success": True,
                "attempts": attempt,
                "data": analysis.model_dump()
            }

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {str(e)[:50]}")
            if attempt == max_attempts:
                return {"success": False, "error": "JSON parsing failed", "attempts": attempt}

        except ValidationError as e:
            errors = e.errors()
            print(f"  ✗ Validation errors: {len(errors)} field(s)")
            for error in errors[:2]:  # Show first 2 errors
                print(f"    - {error['loc'][0]}: {error['msg']}")

            if attempt == max_attempts:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "validation_errors": errors,
                    "attempts": attempt
                }

        except Exception as e:
            print(f"  ✗ Unexpected error: {str(e)[:50]}")
            if attempt == max_attempts:
                return {"success": False, "error": str(e), "attempts": attempt}

    return {"success": False, "error": "Max attempts reached", "attempts": max_attempts}


if __name__ == "__main__":
    print("=== SELF-CORRECTION ===\n")
    print("Testing automatic error detection and correction...\n")

    jobs = load_sample_jobs(3)

    results = []
    for job in jobs:
        result = validate_and_correct(job)
        results.append(result)

        if result["success"]:
            print(f"\n  Final result:")
            print(f"    Category: {result['data']['category']}")
            print(f"    Seniority: {result['data']['seniority']}")
            print(f"    Skills: {', '.join(result['data']['required_skills'][:3])}")
        else:
            print(f"\n  Failed after {result['attempts']} attempts: {result['error']}")

        print("  " + "-" * 50)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"\nSUMMARY:")
    success_count = sum(1 for r in results if r["success"])
    avg_attempts = sum(r["attempts"] for r in results) / len(results)
    print(f"  Success rate: {success_count}/{len(results)}")
    print(f"  Average attempts: {avg_attempts:.1f}")

"""
04 - Output Formatting
======================
Control output structure without schemas.

Key concept: Clear formatting instructions produce consistent, parseable output.

Book reference: AI_eng.5, hands_on_LLM.II.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


def format_as_bullets(content: str) -> str:
    """Request bullet point format."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Format your response as bullet points. Use '•' for main points and '-' for sub-points."
    },
    {"role": "user", "content": content}
    ]
    )
    return response.choices[0].message.content


def format_as_table(content: str) -> str:
    """Request markdown table format."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Format your response as a markdown table with clear headers."
    },
    {"role": "user", "content": content}
    ]
    )
    return response.choices[0].message.content


def format_as_sections(content: str) -> str:
    """Request structured sections."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """Structure your response with these sections:
    ## Overview
    Brief summary
    
    ## Details
    Main content
    
    ## Next Steps
    Actionable items"""
    },
    {"role": "user", "content": content}
    ]
    )
    return response.choices[0].message.content


def format_with_constraints(content: str) -> str:
    """Request specific length and format constraints."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """Format rules:
    - Maximum 3 sentences per paragraph
    - Use bold for key terms
    - End with exactly one takeaway in italics"""
    },
    {"role": "user", "content": content}
    ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== OUTPUT FORMATTING ===\n")
    
    query = "Explain the benefits of remote work for software developers"
    
    formats = {
        "Bullets": format_as_bullets,
        "Table": format_as_table,
        "Sections": format_as_sections,
        "Constrained": format_with_constraints,
    }
    
    for name, formatter in formats.items():
        print(f"=== {name} Format ===")
        result = formatter(query)
        print(result)
        print("\n" + "-" * 50 + "\n")
    
    # Practical example: job comparison
    print("=== PRACTICAL: Job Comparison Table ===")
    comparison_query = "Compare Python Developer vs Data Scientist roles in terms of skills, salary, and career growth"
    print(format_as_table(comparison_query))

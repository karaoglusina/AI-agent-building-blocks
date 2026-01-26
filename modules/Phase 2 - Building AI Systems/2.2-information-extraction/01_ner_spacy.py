"""
01 - NER with spaCy
===================
Extract named entities (ORG, PERSON, LOCATION, etc.) from text.

Key concept: NER identifies real-world objects in text - crucial for information extraction.

Book reference: NLP_cook.5, speach_lang.III.17.3
"""

import spacy

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

# Load spaCy model with NER
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_),
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def group_entities_by_type(entities: list[dict]) -> dict[str, list[str]]:
    """Group entities by their type."""
    grouped = {}
    for ent in entities:
        label = ent["label"]
        if label not in grouped:
            grouped[label] = []
        if ent["text"] not in grouped[label]:  # Deduplicate
            grouped[label].append(ent["text"])
    return grouped


if __name__ == "__main__":
    # Sample text with various entities
    sample = """
    Google is hiring a Senior Engineer in New York. The role pays $150,000 
    and requires 5 years of experience. Contact John Smith at john@google.com.
    The position starts on January 15, 2025.
    """
    
    print("=== SAMPLE TEXT ===")
    print(sample.strip())
    print()
    
    entities = extract_entities(sample)
    print("=== EXTRACTED ENTITIES ===")
    for ent in entities:
        print(f"  {ent['text']:<20} -> {ent['label']:<10} ({ent['description']})")
    print()
    
    # Real job description
    print("=== FROM JOB POSTING ===")
    jobs = load_sample_jobs(1)
    job_entities = extract_entities(jobs[0]["description"][:1000])
    grouped = group_entities_by_type(job_entities)
    
    for label, values in grouped.items():
        print(f"\n{label}:")
        for val in values[:5]:  # First 5 of each type
            print(f"  • {val}")
    
    print("\n=== ENTITY TYPES REFERENCE ===")
    common_types = ["ORG", "PERSON", "GPE", "DATE", "MONEY", "PRODUCT", "SKILL"]
    for t in common_types:
        desc = spacy.explain(t)
        if desc:
            print(f"  {t}: {desc}")

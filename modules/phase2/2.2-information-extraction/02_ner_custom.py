"""
02 - Custom Entity Patterns
===========================
Add custom entity rules and gazetteers to spaCy for domain-specific NER.

Key concept: Standard NER misses domain terms - extend with custom patterns.

Book reference: NLP_cook.5, hands_on_LLM.III.11
"""

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

nlp = spacy.load("en_core_web_sm")


def create_skill_matcher() -> PhraseMatcher:
    """Create a matcher for technical skills (gazetteer approach)."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Skills gazetteer - expand as needed
    skills = [
        # Programming languages
        "python", "java", "javascript", "typescript", "go", "rust", "c++",
        # Frameworks
        "react", "angular", "vue", "django", "fastapi", "spring boot",
        # Data/ML
        "machine learning", "deep learning", "pytorch", "tensorflow",
        "pandas", "numpy", "scikit-learn",
        # Cloud/DevOps
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        # Databases
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    ]
    
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILL", patterns)
    return matcher


def extract_with_custom_entities(text: str, matcher: PhraseMatcher) -> list[dict]:
    """Extract both standard and custom entities."""
    doc = nlp(text)
    
    # Get standard entities
    entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
    
    # Get custom matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        # Avoid overlapping with existing entities
        overlaps = any(
            start < e[3] and end > e[2] for e in entities
        )
        if not overlaps:
            entities.append((span.text, "SKILL", start, end))
    
    return [{"text": t, "label": l} for t, l, s, e in entities]


def add_entity_ruler(nlp_model) -> spacy.Language:
    """Add EntityRuler for pattern-based extraction."""
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    
    patterns = [
        # Job levels
        {"label": "JOB_LEVEL", "pattern": [{"LOWER": {"IN": ["senior", "sr", "junior", "jr", "lead", "principal"]}}]},
        # Certifications
        {"label": "CERT", "pattern": [{"TEXT": {"REGEX": "AWS|GCP|Azure"}}, {"LOWER": "certified", "OP": "?"}]},
        # Years of experience
        {"label": "EXPERIENCE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs", "year"]}}]},
    ]
    
    ruler.add_patterns(patterns)
    return nlp_model


if __name__ == "__main__":
    # Sample job description
    text = """
    We're looking for a Senior Python Developer with 5+ years experience.
    Required skills: Python, Django, PostgreSQL, Docker, and AWS.
    Experience with machine learning (PyTorch or TensorFlow) is a plus.
    AWS certified candidates preferred.
    """
    
    print("=== STANDARD NER ONLY ===")
    doc = nlp(text)
    for ent in doc.ents:
        print(f"  {ent.text:<20} -> {ent.label_}")
    print()
    
    print("=== WITH CUSTOM SKILL MATCHER ===")
    skill_matcher = create_skill_matcher()
    entities = extract_with_custom_entities(text, skill_matcher)
    for ent in entities:
        print(f"  {ent['text']:<20} -> {ent['label']}")
    print()
    
    # Real job posting
    print("=== SKILLS FROM JOB POSTING ===")
    jobs = load_sample_jobs(1)
    job_entities = extract_with_custom_entities(jobs[0]["description"], skill_matcher)
    skills = set(e["text"] for e in job_entities if e["label"] == "SKILL")
    print(f"  Found {len(skills)} skills: {', '.join(sorted(skills))}")

"""
04 - Noun Chunks & Grammar
==========================
Extract noun phrases and analyze grammatical structure.

Key concept: Noun chunks capture meaningful phrases, not just individual words.

Book reference: NLP_cook.2, speach_lang.III.19
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import spacy
except ImportError:
    MISSING_DEPENDENCIES.append('spacy')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path

nlp = spacy.load("en_core_web_sm")


def extract_noun_chunks(text: str) -> list[dict]:
    """Extract noun phrases with their grammatical roles."""
    doc = nlp(text)
    chunks = []
    for chunk in doc.noun_chunks:
        chunks.append({
            "text": chunk.text,
            "root": chunk.root.text,        # Main noun
            "root_dep": chunk.root.dep_,    # Grammatical role
            "root_head": chunk.root.head.text,  # What it relates to
        })
    return chunks


def get_subject_verb_object(text: str) -> list[dict]:
    """Extract subject-verb-object triples from sentences."""
    doc = nlp(text)
    triples = []
    
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            triple = {"verb": token.text, "subject": None, "object": None}
            
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    triple["subject"] = child.text
                elif child.dep_ in ("dobj", "pobj", "attr"):
                    triple["object"] = child.text
            
            if triple["subject"] or triple["object"]:
                triples.append(triple)
    
    return triples


def get_dependency_tree(text: str) -> list[tuple]:
    """Get the dependency parse tree."""
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]


if __name__ == "__main__":
    # Noun chunks extraction
    text = """
    The experienced Python developer will build scalable web applications.
    Strong problem-solving skills and excellent communication are required.
    """
    
    print("=== NOUN CHUNKS ===")
    chunks = extract_noun_chunks(text)
    print(f"{'Chunk':<35} {'Root':<15} {'Role':<10} {'Head'}")
    print("-" * 70)
    for chunk in chunks:
        print(f"{chunk['text']:<35} {chunk['root']:<15} {chunk['root_dep']:<10} {chunk['root_head']}")
    print()
    
    # Subject-Verb-Object extraction
    print("=== SUBJECT-VERB-OBJECT ===")
    sentences = [
        "The company develops innovative software.",
        "We are seeking talented engineers.",
        "The role requires Python experience."]
    for sent in sentences:
        triples = get_subject_verb_object(sent)
        for t in triples:
            print(f"  {t['subject']} --[{t['verb']}]--> {t['object']}")
    print()
    
    # Job description analysis
    print("=== KEY PHRASES FROM JOB POSTING ===")
    jobs = load_sample_jobs(1)
    chunks = extract_noun_chunks(jobs[0]["description"][:800])
    
    # Get unique noun chunks, sorted by length (longer = more specific)
    unique_chunks = sorted(set(c["text"].lower() for c in chunks), 
                          key=len, reverse=True)
    print("Top noun phrases:")
    for chunk in unique_chunks[:10]:
        print(f"  • {chunk}")

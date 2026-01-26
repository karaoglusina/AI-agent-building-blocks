"""
10 - Fuzzy String Matching
==========================
Handle typos and variations with edit distance and fuzzy matching.

Key concept: Fuzzy matching finds similar strings despite spelling differences.

Book reference: NLP_cook.5, speach_lang.I.2.8
"""

from rapidfuzz import fuzz, process

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs


# Standard skill names for normalization
CANONICAL_SKILLS = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
    "React", "Angular", "Vue.js", "Node.js", "Django", "FastAPI", "Flask",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
]


def simple_ratio(s1: str, s2: str) -> int:
    """Basic similarity ratio (0-100)."""
    return fuzz.ratio(s1.lower(), s2.lower())


def partial_ratio(s1: str, s2: str) -> int:
    """Partial match - good for substrings."""
    return fuzz.partial_ratio(s1.lower(), s2.lower())


def token_sort_ratio(s1: str, s2: str) -> int:
    """Order-independent comparison."""
    return fuzz.token_sort_ratio(s1.lower(), s2.lower())


def find_best_match(query: str, choices: list[str], threshold: int = 70) -> tuple[str, int] | None:
    """Find the best matching string from choices."""
    result = process.extractOne(query, choices, score_cutoff=threshold)
    return result if result else None


def normalize_skill(skill: str) -> str:
    """Normalize a skill name to its canonical form."""
    match = find_best_match(skill, CANONICAL_SKILLS, threshold=75)
    return match[0] if match else skill


def deduplicate_skills(skills: list[str], threshold: int = 80) -> list[str]:
    """Remove duplicate skills based on fuzzy matching."""
    unique = []
    for skill in skills:
        # Check if similar to any existing
        is_duplicate = any(
            fuzz.ratio(skill.lower(), existing.lower()) >= threshold
            for existing in unique
        )
        if not is_duplicate:
            unique.append(skill)
    return unique


if __name__ == "__main__":
    print("=== FUZZY MATCHING EXAMPLES ===\n")
    
    # Different matching strategies
    pairs = [
        ("Python", "python"),
        ("JavaScript", "Java Script"),
        ("PostgreSQL", "Postgres"),
        ("Machine Learning", "ML"),
        ("Kubernetes", "K8s"),
    ]
    
    print(f"{'Pair':<35} {'Simple':<8} {'Partial':<8} {'Token':<8}")
    print("-" * 60)
    for s1, s2 in pairs:
        simple = simple_ratio(s1, s2)
        partial = partial_ratio(s1, s2)
        token = token_sort_ratio(s1, s2)
        print(f"{s1} vs {s2:<15} {simple:<8} {partial:<8} {token:<8}")
    print()
    
    # Skill normalization
    print("=== SKILL NORMALIZATION ===")
    raw_skills = ["python3", "JS", "React.js", "postgres", "k8s", "pytorch", "aws cloud"]
    print(f"Input: {raw_skills}")
    print("Normalized:")
    for skill in raw_skills:
        normalized = normalize_skill(skill)
        print(f"  {skill:<15} -> {normalized}")
    print()
    
    # Deduplication
    print("=== DEDUPLICATION ===")
    duplicated = ["Python", "python", "PYTHON", "React", "React.js", "ReactJS",
                  "Machine Learning", "ML", "PostgreSQL", "Postgres"]
    unique = deduplicate_skills(duplicated)
    print(f"Input ({len(duplicated)}): {duplicated}")
    print(f"Deduped ({len(unique)}): {unique}")
    print()
    
    # Real job matching
    print("=== FIND MATCHING JOBS ===")
    jobs = load_sample_jobs(10)
    query = "Senior Python Developer"
    
    print(f"Query: '{query}'\n")
    titles = [job["title"] for job in jobs]
    matches = process.extract(query, titles, limit=5)
    
    for title, score, idx in matches:
        print(f"  {score:3}% - {title}")

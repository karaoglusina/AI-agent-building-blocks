"""
02 - Relation Extraction
=========================
Extract relationships between entities from text.

Key concept: Relation extraction identifies semantic relationships between entities (e.g., "works for", "located in", "invented by"), enabling knowledge graph construction.

Book reference: speach_lang.III.20
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def relation_extraction_intro():
    """Introduce relation extraction."""
    print("=== RELATION EXTRACTION ===\n")

    print("What is relation extraction?")
    print("  Identifying and classifying semantic relationships between entities in text.\n")

    print("Example:")
    print("  Text: 'Steve Jobs founded Apple in 1976'")
    print("  Entities: [Steve Jobs (PERSON)], [Apple (ORG)], [1976 (DATE)]")
    print("  Relations:")
    print("    - (Steve Jobs, FOUNDED, Apple)")
    print("    - (Apple, FOUNDED_IN, 1976)\n")

    print("Why relation extraction?")
    print("  ✓ Build knowledge graphs from text")
    print("  ✓ Extract structured facts from documents")
    print("  ✓ Answer complex queries")
    print("  ✓ Discover connections between entities")
    print("  ✓ Enhance search and recommendation systems")
    print("  ✓ Support fact-checking and verification")


def approaches_overview():
    """Overview of relation extraction approaches."""
    print("\n" + "=" * 70)
    print("=== APPROACHES TO RELATION EXTRACTION ===\n")

    print("1. Rule-based")
    print("   Use hand-crafted patterns and linguistic rules")
    print("   Example: 'X founded Y' → (X, FOUNDED, Y)")
    print("   ✓ High precision, interpretable")
    print("   ✗ Low coverage, labor-intensive\n")

    print("2. Supervised Learning")
    print("   Train classifier on labeled (entity1, relation, entity2) examples")
    print("   ✓ Good accuracy with enough training data")
    print("   ✗ Requires labeled data, limited to known relations\n")

    print("3. Distant Supervision")
    print("   Use knowledge base to automatically label training data")
    print("   ✓ Scalable, no manual labeling")
    print("   ✗ Noisy labels, may miss implicit relations\n")

    print("4. LLM-based")
    print("   Use LLMs (GPT-4, Claude) to extract relations with prompts")
    print("   ✓ Zero-shot, flexible, handles complex relations")
    print("   ✗ Slower, more expensive")


def rule_based_example():
    """Show rule-based relation extraction."""
    print("\n" + "=" * 70)
    print("=== RULE-BASED EXTRACTION ===\n")

    code = '''import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

def extract_founder_relations(text):
    """Extract 'founded' relationships using patterns."""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Pattern: PERSON + founded/created/started + ORG
    pattern1 = [
        {"ENT_TYPE": "PERSON"},
        {"LOWER": {"IN": ["founded", "created", "started", "established"]}},
        {"ENT_TYPE": "ORG"}
    ]

    # Pattern: ORG + was founded by + PERSON
    pattern2 = [
        {"ENT_TYPE": "ORG"},
        {"LOWER": "was"},
        {"LOWER": {"IN": ["founded", "created", "started"]}},
        {"LOWER": "by"},
        {"ENT_TYPE": "PERSON"}
    ]

    matcher.add("FOUNDER", [pattern1, pattern2])

    matches = matcher(doc)
    relations = []

    for match_id, start, end in matches:
        span = doc[start:end]
        # Extract entities
        entities = [ent for ent in span.ents]
        if len(entities) >= 2:
            # Determine order based on pattern
            if entities[0].label_ == "PERSON":
                founder = entities[0].text
                company = entities[1].text
            else:
                company = entities[0].text
                founder = entities[1].text

            relations.append((founder, "FOUNDED", company))

    return relations

# Test
texts = [
    "Steve Jobs founded Apple in 1976",
    "Microsoft was founded by Bill Gates and Paul Allen",
    "Larry Page and Sergey Brin created Google"
]

for text in texts:
    relations = extract_founder_relations(text)
    print(f"Text: {text}")
    for rel in relations:
        print(f"  {rel}")
    print()
'''

    print(code)


def dependency_based_extraction():
    """Show dependency-based relation extraction."""
    print("\n" + "=" * 70)
    print("=== DEPENDENCY-BASED EXTRACTION ===\n")

    code = '''def extract_employment_relations(text):
    """Extract employment relationships using dependency parsing."""
    doc = nlp(text)
    relations = []

    for token in doc:
        # Look for verbs indicating employment
        if token.lemma_ in ["work", "employ", "hire"]:
            subject = None
            organization = None

            # Find subject (employee)
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    if child.ent_type_ == "PERSON":
                        subject = child.text

            # Find object or prepositional phrase (employer)
            for child in token.children:
                if child.dep_ == "prep" and child.text in ["for", "at", "with"]:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            if grandchild.ent_type_ == "ORG":
                                organization = grandchild.text

            if subject and organization:
                relations.append((subject, "WORKS_FOR", organization))

    return relations

# Test
texts = [
    "John Smith works for Google",
    "Alice was hired by Microsoft",
    "Bob is employed at Amazon"
]

for text in texts:
    rels = extract_employment_relations(text)
    for rel in rels:
        print(f"{rel[0]} {rel[1]} {rel[2]}")
'''

    print(code)


def llm_based_extraction():
    """Show LLM-based relation extraction."""
    print("\n" + "=" * 70)
    print("=== LLM-BASED EXTRACTION ===\n")

    code = '''from openai import OpenAI

client = OpenAI()

def extract_relations_with_llm(text):
    """Extract relations using GPT-4."""

    prompt = f"""Extract all relationships from the following text.
For each relationship, identify:
- Entity 1 (with type)
- Relation
- Entity 2 (with type)

Format: (Entity1, Relation, Entity2)

Text: {text}

Relations:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Parse response
    relations_text = response.choices[0].message.content
    # ... parse the structured output ...

    return relations_text

# Test
text = """
Elon Musk is the CEO of Tesla and SpaceX. He was born in South Africa
in 1971. Tesla produces electric vehicles and is headquartered in Austin, Texas.
"""

relations = extract_relations_with_llm(text)
print(relations)

# Expected output (structured):
# (Elon Musk [PERSON], CEO_OF, Tesla [ORG])
# (Elon Musk [PERSON], CEO_OF, SpaceX [ORG])
# (Elon Musk [PERSON], BORN_IN, South Africa [LOC])
# (Elon Musk [PERSON], BORN_IN, 1971 [DATE])
# (Tesla [ORG], PRODUCES, electric vehicles [PRODUCT])
# (Tesla [ORG], HEADQUARTERED_IN, Austin [LOC])
'''

    print(code)


def common_relation_types():
    """List common relation types."""
    print("\n" + "=" * 70)
    print("=== COMMON RELATION TYPES ===\n")

    relations = [
        ("Organizational", "WORKS_FOR, CEO_OF, FOUNDED, MEMBER_OF"),
        ("Geographic", "LOCATED_IN, BORN_IN, HEADQUARTERED_IN, CAPITAL_OF"),
        ("Family", "PARENT_OF, MARRIED_TO, SIBLING_OF, CHILD_OF"),
        ("Temporal", "FOUNDED_IN, OCCURRED_ON, DIED_IN, STARTED_IN"),
        ("Product", "PRODUCES, MANUFACTURES, INVENTED, DEVELOPS"),
        ("Academic", "EDUCATED_AT, PROFESSOR_AT, ALUMNUS_OF, STUDIED"),
        ("Ownership", "OWNS, ACQUIRED, SUBSIDIARY_OF, PARENT_COMPANY"),
        ("Causal", "CAUSES, PREVENTS, TREATS, RESULTS_IN"),
    ]

    print("Category        Example Relations")
    print("-" * 70)
    for category, examples in relations:
        print(f"{category:20}{examples}")


def building_knowledge_graph():
    """Show how to build knowledge graph from extracted relations."""
    print("\n" + "=" * 70)
    print("=== BUILDING KNOWLEDGE GRAPH ===\n")

    code = '''class KnowledgeGraph:
    """Simple knowledge graph from extracted relations."""

    def __init__(self):
        self.entities = {}
        self.relations = []

    def add_relation(self, entity1, relation, entity2):
        """Add a relation to the graph."""
        # Add entities
        if entity1 not in self.entities:
            self.entities[entity1] = {"name": entity1, "relations": []}
        if entity2 not in self.entities:
            self.entities[entity2] = {"name": entity2, "relations": []}

        # Add relation
        rel = {
            "subject": entity1,
            "predicate": relation,
            "object": entity2
        }
        self.relations.append(rel)
        self.entities[entity1]["relations"].append(rel)

    def query(self, entity):
        """Query all relations for an entity."""
        if entity in self.entities:
            return self.entities[entity]["relations"]
        return []

    def find_path(self, entity1, entity2):
        """Find relationship path between two entities (BFS)."""
        # Implementation of graph search...
        pass

# Build knowledge graph from text
kg = KnowledgeGraph()

texts = [
    "Steve Jobs founded Apple",
    "Tim Cook is CEO of Apple",
    "Apple is headquartered in Cupertino"
]

for text in texts:
    relations = extract_relations(text)  # Using any extraction method
    for subj, pred, obj in relations:
        kg.add_relation(subj, pred, obj)

# Query the graph
apple_relations = kg.query("Apple")
for rel in apple_relations:
    print(f"{rel['subject']} --{rel['predicate']}--> {rel['object']}")

# Output:
# Steve Jobs --FOUNDED--> Apple
# Tim Cook --CEO_OF--> Apple
# Apple --HEADQUARTERED_IN--> Cupertino
'''

    print(code)


def evaluation_metrics():
    """Explain evaluation metrics for relation extraction."""
    print("\n" + "=" * 70)
    print("=== EVALUATION METRICS ===\n")

    print("Precision:")
    print("  Of extracted relations, how many are correct?")
    print("  Precision = Correct Extractions / Total Extractions\n")

    print("Recall:")
    print("  Of all true relations, how many did we extract?")
    print("  Recall = Correct Extractions / Total True Relations\n")

    print("F1 Score:")
    print("  Harmonic mean of precision and recall")
    print("  F1 = 2 × (Precision × Recall) / (Precision + Recall)\n")

    print("Example:")
    print("  True relations: 10")
    print("  Extracted: 8")
    print("  Correct: 6")
    print("  Precision: 6/8 = 0.75 (75%)")
    print("  Recall: 6/10 = 0.60 (60%)")
    print("  F1: 2×(0.75×0.60)/(0.75+0.60) = 0.67")


def real_world_applications():
    """Show real-world applications."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD APPLICATIONS ===\n")

    print("1. Knowledge Base Construction")
    print("   Extract facts from Wikipedia → Build knowledge graph")
    print("   Example: DBpedia, Wikidata\n")

    print("2. Biomedical Research")
    print("   Extract drug-disease relationships from papers")
    print("   '(Aspirin, TREATS, headache)'\n")

    print("3. Financial Analysis")
    print("   Extract corporate relationships from news")
    print("   '(Company A, ACQUIRED, Company B)'\n")

    print("4. Intelligence & Security")
    print("   Extract person-organization connections")
    print("   Map networks and relationships\n")

    print("5. Customer Service")
    print("   Extract product-problem relationships")
    print("   '(Product X, HAS_ISSUE, heating problem)'")


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Start with NER", "Good entity recognition is foundation"),
        ("Combine approaches", "Use rules + ML + LLM for best coverage"),
        ("Validate extractions", "Filter low-confidence relations"),
        ("Handle negations", "Don't extract negated relations"),
        ("Normalize relations", "Standardize relation names"),
        ("Track provenance", "Record source text for each relation"),
        ("Iterative refinement", "Review errors, add patterns"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def challenges_and_solutions():
    """Discuss challenges."""
    print("=" * 70)
    print("=== CHALLENGES AND SOLUTIONS ===\n")

    challenges = [
        ("Ambiguity",
         "Same text, multiple interpretations",
         "Use context, confidence scores, human review"),

        ("Implicit relations",
         "'CEO of company' implies WORKS_FOR",
         "Add inference rules, use LLMs"),

        ("Long-distance dependencies",
         "Entities far apart in text",
         "Use better parsing, coreference resolution"),

        ("Domain-specific relations",
         "Need custom relation types",
         "Define domain ontology, fine-tune models"),

        ("Noisy text",
         "Social media, OCR errors",
         "Robust NER, fuzzy matching"),
    ]

    for challenge, description, solution in challenges:
        print(f"Challenge: {challenge}")
        print(f"  Problem: {description}")
        print(f"  Solution: {solution}\n")


if __name__ == "__main__":
    relation_extraction_intro()
    approaches_overview()
    rule_based_example()
    dependency_based_extraction()
    llm_based_extraction()
    common_relation_types()
    building_knowledge_graph()
    evaluation_metrics()
    real_world_applications()
    best_practices()
    challenges_and_solutions()

    print("\n" + "=" * 70)
    print("\nKey insight: Relation extraction turns text into structured knowledge!")
    print("Combine rules, ML, and LLMs for comprehensive extraction")

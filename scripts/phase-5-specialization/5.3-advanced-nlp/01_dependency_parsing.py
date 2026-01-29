"""
01 - Dependency Parsing
========================
Extract grammatical structure with spaCy.

Key concept: Dependency parsing reveals grammatical relationships between words, useful for extracting structured information and understanding sentence meaning.

Book reference: NLP_cook.2, speach_lang.III.19
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def dependency_parsing_intro():
    """Introduce dependency parsing."""
    print("=== DEPENDENCY PARSING ===\n")

    print("What is dependency parsing?")
    print("  Analyzing the grammatical structure of a sentence by identifying")
    print("  relationships between words (subject, object, modifier, etc.)\n")

    print("Example sentence: 'The cat chased the mouse'")
    print("  Dependencies:")
    print("    chased (ROOT)")
    print("    ├── cat (subject)")
    print("    │   └── The (determiner)")
    print("    └── mouse (object)")
    print("        └── the (determiner)\n")

    print("Why use dependency parsing?")
    print("  ✓ Extract subject-verb-object relationships")
    print("  ✓ Find modifiers and attributes")
    print("  ✓ Understand sentence structure")
    print("  ✓ Extract facts from text")
    print("  ✓ Build knowledge graphs")
    print("  ✓ Improve information extraction")


def basic_example():
    """Show basic dependency parsing example."""
    print("\n" + "=" * 70)
    print("=== BASIC EXAMPLE (CONCEPTUAL) ===\n")

    code = '''import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Parse sentence
doc = nlp("The cat sat on the mat")

# Print dependencies
print("Token       Dep      Head       Children")
print("-" * 50)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:12}{token.dep_:10}{token.head.text:12}{children}")

# Output:
# Token       Dep      Head       Children
# --------------------------------------------------
# The         det      cat        []
# cat         nsubj    sat        ['The']
# sat         ROOT     sat        ['cat', 'on']
# on          prep     sat        ['mat']
# the         det      mat        []
# mat         pobj     on         ['the']
'''

    print(code)


def common_dependencies():
    """Explain common dependency relations."""
    print("\n" + "=" * 70)
    print("=== COMMON DEPENDENCY RELATIONS ===\n")

    relations = [
        ("nsubj", "Nominal subject", "She runs → She is subject of runs"),
        ("dobj", "Direct object", "read book → book is object of read"),
        ("amod", "Adjectival modifier", "red car → red modifies car"),
        ("prep", "Prepositional modifier", "on table → on is preposition"),
        ("pobj", "Object of preposition", "on table → table is object of on"),
        ("det", "Determiner", "the cat → the determines cat"),
        ("aux", "Auxiliary verb", "will go → will is auxiliary"),
        ("ROOT", "Root of sentence", "Main verb or predicate"),
        ("compound", "Compound", "phone book → compound noun"),
        ("advmod", "Adverbial modifier", "runs quickly → quickly modifies runs")]

    print("Relation    Type                     Example")
    print("-" * 70)
    for rel, type_desc, example in relations:
        print(f"{rel:12}{type_desc:25}{example}")


def extract_subjects_and_objects():
    """Show how to extract subjects and objects."""
    print("\n" + "=" * 70)
    print("=== EXTRACTING SUBJECTS AND OBJECTS ===\n")

    code = '''import spacy

nlp = spacy.load("en_core_web_sm")

def extract_subject_verb_object(text):
    """Extract (subject, verb, object) triples."""
    doc = nlp(text)
    triples = []

    for token in doc:
        # Find verbs
        if token.pos_ == "VERB":
            subject = None
            obj = None

            # Find subject
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child.text

                # Find object
                if child.dep_ in ["dobj", "attr", "prep"]:
                    if child.dep_ == "prep":
                        # Get object of preposition
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = f"{child.text} {grandchild.text}"
                    else:
                        obj = child.text

            if subject and obj:
                triples.append((subject, token.text, obj))

    return triples

# Test
sentences = [
    "The company released a new product",
    "Scientists discovered a cure for cancer",
    "The cat jumped on the table"
]

for sent in sentences:
    triples = extract_subject_verb_object(sent)
    print(f"Sentence: {sent}")
    for subj, verb, obj in triples:
        print(f"  ({subj}, {verb}, {obj})")
    print()

# Output:
# Sentence: The company released a new product
#   (company, released, product)
# Sentence: Scientists discovered a cure for cancer
#   (Scientists, discovered, cure)
# Sentence: The cat jumped on the table
#   (cat, jumped, on table)
'''

    print(code)


def extract_noun_phrases():
    """Show how to extract noun phrases and their modifiers."""
    print("\n" + "=" * 70)
    print("=== EXTRACTING NOUN PHRASES ===\n")

    code = '''def extract_noun_phrases_with_modifiers(text):
    """Extract noun phrases with their adjective modifiers."""
    doc = nlp(text)
    phrases = []

    for chunk in doc.noun_chunks:
        # Get modifiers
        modifiers = [token.text for token in chunk if token.pos_ == "ADJ"]

        # Get head noun
        head = chunk.root.text

        phrases.append({
            "full_phrase": chunk.text,
            "head": head,
            "modifiers": modifiers
        })

    return phrases

# Test
text = "The quick brown fox jumped over the lazy dog"
phrases = extract_noun_phrases_with_modifiers(text)

for phrase in phrases:
    print(f"Phrase: {phrase['full_phrase']}")
    print(f"  Head: {phrase['head']}")
    print(f"  Modifiers: {phrase['modifiers']}")
    print()

# Output:
# Phrase: The quick brown fox
#   Head: fox
#   Modifiers: ['quick', 'brown']
# Phrase: the lazy dog
#   Head: dog
#   Modifiers: ['lazy']
'''

    print(code)


def visualize_dependencies():
    """Show how to visualize dependency trees."""
    print("\n" + "=" * 70)
    print("=== VISUALIZING DEPENDENCIES ===\n")

    code = '''from spacy import displacy

# Parse sentence
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Render in browser
displacy.serve(doc, style="dep")

# Or save to SVG file
svg = displacy.render(doc, style="dep", jupyter=False)
with open("dependency_tree.svg", "w") as f:
    f.write(svg)

# Print text-based tree
def print_tree(token, depth=0):
    """Print dependency tree recursively."""
    print("  " * depth + f"{token.text} ({token.dep_})")
    for child in token.children:
        print_tree(child, depth + 1)

# Print tree for root
for token in doc:
    if token.dep_ == "ROOT":
        print_tree(token)

# Output:
# looking (ROOT)
#   Apple (nsubj)
#   is (aux)
#   at (prep)
#     buying (pcomp)
#       startup (dobj)
#         U.K. (compound)
#       for (prep)
#         $ (quantmod)
#         billion (pobj)
#           1 (compound)
'''

    print(code)


def real_world_applications():
    """Show real-world applications."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD APPLICATIONS ===\n")

    print("1. Question Answering")
    print("   Extract answer patterns based on question structure")
    print("   Q: 'Who founded Microsoft?' → Look for (X, founded, Microsoft)\n")

    print("2. Fact Extraction")
    print("   Build knowledge graphs from text")
    print("   'Tesla produces electric cars' → (Tesla, produces, cars)\n")

    print("3. Text Simplification")
    print("   Identify complex clauses and subordinate relationships")
    print("   Simplify nested structures\n")

    print("4. Sentiment Analysis")
    print("   Find opinion targets and sentiment words")
    print("   'The food was amazing' → (food, was, amazing [positive])\n")

    print("5. Information Extraction")
    print("   Extract structured data from unstructured text")
    print("   'CEO John Smith announced...' → (John Smith, role: CEO)")


def advanced_patterns():
    """Show advanced dependency patterns."""
    print("\n" + "=" * 70)
    print("=== ADVANCED PATTERNS ===\n")

    code = '''def extract_passive_voice(text):
    """Find passive voice constructions."""
    doc = nlp(text)
    passives = []

    for token in doc:
        # Look for passive subjects (nsubjpass)
        if token.dep_ == "nsubjpass":
            # Find the main verb
            verb = token.head
            # Find the agent (by phrase)
            agent = None
            for child in verb.children:
                if child.dep_ == "agent":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            agent = grandchild.text

            passives.append({
                "subject": token.text,
                "verb": verb.text,
                "agent": agent
            })

    return passives

# Test
text = "The ball was thrown by John"
passives = extract_passive_voice(text)
for p in passives:
    print(f"{p['subject']} was {p['verb']} by {p['agent']}")
# Output: ball was thrown by John


def find_negations(text):
    """Find negated statements."""
    doc = nlp(text)
    negations = []

    for token in doc:
        # Look for negation markers
        if token.dep_ == "neg":
            negated_word = token.head
            negations.append({
                "negated": negated_word.text,
                "full_context": negated_word.sent.text
            })

    return negations

# Test
text = "I do not like spam. She never eats meat."
for neg in find_negations(text):
    print(f"Negated: {neg['negated']} in '{neg['full_context']}'")
# Output:
# Negated: like in 'I do not like spam.'
# Negated: eats in 'She never eats meat.'
'''

    print(code)


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Choose right model", "en_core_web_sm (fast) vs en_core_web_lg (accurate)"),
        ("Batch processing", "Process multiple docs with nlp.pipe()"),
        ("Disable unused pipes", "Speed up with disable=['ner', 'textcat']"),
        ("Custom rules", "Add domain-specific dependency patterns"),
        ("Validate results", "Dependencies not always perfect, add error handling"),
        ("Combine with other tools", "Use with NER, coreference, etc.")]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def limitations():
    """Discuss limitations."""
    print("=" * 70)
    print("=== LIMITATIONS ===\n")

    print("Challenges:")
    print("  • Errors on complex or unusual sentence structures")
    print("  • Domain-specific language may confuse parser")
    print("  • Long sentences harder to parse accurately")
    print("  • Ambiguous constructions may be misparsed")
    print("  • Performance degrades on noisy text (social media, etc.)\n")

    print("Solutions:")
    print("  → Use larger models (web_lg) for better accuracy")
    print("  → Clean and normalize text before parsing")
    print("  → Add custom rules for domain-specific patterns")
    print("  → Combine with other NLP techniques")
    print("  → Validate and filter low-confidence parses")


if __name__ == "__main__":
    dependency_parsing_intro()
    basic_example()
    common_dependencies()
    extract_subjects_and_objects()
    extract_noun_phrases()
    visualize_dependencies()
    real_world_applications()
    advanced_patterns()
    best_practices()
    limitations()

    print("\n" + "=" * 70)
    print("\nKey insight: Dependency parsing unlocks sentence structure!")
    print("Extract facts, relationships, and meaning from unstructured text")

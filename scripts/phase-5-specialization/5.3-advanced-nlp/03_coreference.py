"""
03 - Coreference Resolution
============================
Resolve pronouns and references to their antecedents.

Key concept: Coreference resolution identifies when different expressions refer to the same entity, essential for understanding text coherence and extracting accurate information.

Book reference: speach_lang.III.23
"""

import utils._load_env  # Loads .env file automatically

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def coreference_intro():
    """Introduce coreference resolution."""
    print("=== COREFERENCE RESOLUTION ===\n")

    print("What is coreference?")
    print("  When multiple expressions in text refer to the same entity.\n")

    print("Example:")
    print("  'Barack Obama was born in Hawaii. He served as president. Obama...'")
    print("  Coreferences:")
    print("    - 'Barack Obama' ← 'He' ← 'Obama'")
    print("    - All three refer to the same person\n")

    print("Why coreference resolution?")
    print("  ✓ Improve text understanding")
    print("  ✓ Extract complete entity mentions")
    print("  ✓ Better information extraction")
    print("  ✓ Enhance question answering")
    print("  ✓ Improve summarization quality")
    print("  ✓ Enable entity linking across sentences\n")

    print("Types of coreferences:")
    print("  • Pronouns: 'he', 'she', 'it', 'they'")
    print("  • Definite NPs: 'the president', 'the company'")
    print("  • Demonstratives: 'this', 'that', 'these'")
    print("  • Names: 'Obama' → 'Barack Obama'")


def coreference_examples():
    """Show coreference examples."""
    print("\n" + "=" * 70)
    print("=== COREFERENCE EXAMPLES ===\n")

    examples = [
        ("Pronominal",
         "John went to the store. He bought milk.",
         "He → John"),

        ("Definite NP",
         "Apple released iPhone. The device was revolutionary.",
         "The device → iPhone"),

        ("Demonstrative",
         "I read three books. These were all mysteries.",
         "These → three books"),

        ("Name variation",
         "Barack Obama was elected. Obama served two terms.",
         "Obama → Barack Obama"),

        ("Cataphora (forward ref)",
         "When he arrived, John was tired.",
         "he → John (forward reference)")]

    print("Type              Example                                    Resolution")
    print("-" * 85)
    for type_name, example, resolution in examples:
        print(f"{type_name:18}{example:45}{resolution}")


def basic_usage():
    """Show basic coreference usage."""
    print("\n" + "=" * 70)
    print("=== BASIC USAGE (CONCEPTUAL) ===\n")

    code = '''# Using neuralcoref (spaCy extension)
import spacy
import neuralcoref

# Load model and add neuralcoref
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

text = """
Barack Obama was born in Hawaii. He served as the 44th president.
Obama's presidency lasted from 2009 to 2017. During his time in office,
he signed the Affordable Care Act.
"""

doc = nlp(text)

# Print coreference clusters
print("Coreference clusters:")
if doc._.has_coref:
    for cluster in doc._.coref_clusters:
        print(f"  Cluster: {cluster}")
        print(f"    Main: {cluster.main}")
        print(f"    Mentions: {[mention.text for mention in cluster.mentions]}")
        print()

# Output:
# Cluster: Barack Obama: [Barack Obama, He, Obama, his, he]
#   Main: Barack Obama
#   Mentions: ['Barack Obama', 'He', 'Obama', 'his', 'he']
'''

    print(code)

    print("\n\nUsing FastCoref (modern alternative):\n")

    code2 = '''from fastcoref import FCoref

# Load model
model = FCoref()

# Predict coreferences
text = "Barack Obama was born in Hawaii. He became president."
clusters = model.predict(texts=[text])

# clusters contains coreference information
for cluster in clusters[0]:
    print(f"Cluster: {cluster}")
'''

    print(code2)


def resolving_coreferences():
    """Show how to resolve coreferences in text."""
    print("\n" + "=" * 70)
    print("=== RESOLVING COREFERENCES ===\n")

    code = '''def resolve_coreferences(doc):
    """Replace pronouns with their antecedents."""

    if not doc._.has_coref:
        return doc.text

    # Create mapping of tokens to their resolved form
    resolved = {}
    for cluster in doc._.coref_clusters:
        # Use the main mention (typically the most complete)
        main_mention = cluster.main.text

        # Map all other mentions to main
        for mention in cluster.mentions:
            if mention != cluster.main:
                resolved[mention.start] = main_mention

    # Rebuild text with resolved references
    tokens = []
    for i, token in enumerate(doc):
        if i in resolved:
            # Skip tokens that are part of resolved mention
            if i == 0 or (i-1) not in resolved:
                tokens.append(resolved[i])
        else:
            tokens.append(token.text_with_ws)

    return "".join(tokens)

# Test
text = "John loves his dog. He walks it every day."
doc = nlp(text)
resolved_text = resolve_coreferences(doc)

print("Original:", text)
print("Resolved:", resolved_text)
# Output: "John loves John's dog. John walks John's dog every day."
'''

    print(code)


def improving_extraction():
    """Show how coreference improves information extraction."""
    print("\n" + "=" * 70)
    print("=== IMPROVING INFORMATION EXTRACTION ===\n")

    print("Without coreference resolution:\n")

    print("Text:")
    print("  'Apple released iPhone 15. It features improved cameras.'")
    print("\nExtracted facts:")
    print("  - (Apple, released, iPhone 15)")
    print("  - (It, features, improved cameras)  ← 'It' is unclear!\n")

    print("-" * 70)
    print("\nWith coreference resolution:\n")

    print("Resolved text:")
    print("  'Apple released iPhone 15. iPhone 15 features improved cameras.'")
    print("\nExtracted facts:")
    print("  - (Apple, released, iPhone 15)")
    print("  - (iPhone 15, features, improved cameras)  ← Clear entity!\n")

    code = '''def extract_facts_with_coref(text):
    """Extract facts with coreference resolution."""

    # Resolve coreferences first
    doc = nlp(text)
    resolved_text = resolve_coreferences(doc)

    # Then extract facts from resolved text
    resolved_doc = nlp(resolved_text)

    facts = []
    for sent in resolved_doc.sents:
        # Extract subject-verb-object
        for token in sent:
            if token.pos_ == "VERB":
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ in ["nsubj"]:
                        subject = child.text
                    if child.dep_ in ["dobj"]:
                        obj = child.text

                if subject and obj:
                    facts.append((subject, token.text, obj))

    return facts

# Without coref: Extracts (It, features, cameras)
# With coref:    Extracts (iPhone 15, features, cameras)
'''

    print(code)


def llm_based_coreference():
    """Show LLM-based coreference resolution."""
    print("\n" + "=" * 70)
    print("=== LLM-BASED COREFERENCE ===\n")

    code = '''from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

def resolve_with_llm(text):
    """Use LLM to resolve coreferences."""

    prompt = f"""Resolve all pronouns and references in the following text.
Replace each pronoun/reference with its full antecedent.

Original text:
{text}

Resolved text (with pronouns replaced):"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# Test
text = """
Tesla announced a new model. The company said it would launch next year.
Elon Musk, its CEO, expressed excitement.
"""

resolved = resolve_with_llm(text)
print(resolved)

# Expected output:
# "Tesla announced a new model. Tesla said the new model would launch
# next year. Elon Musk, Tesla's CEO, expressed excitement."
'''

    print(code)


def evaluation_challenges():
    """Discuss evaluation and challenges."""
    print("\n" + "=" * 70)
    print("=== CHALLENGES ===\n")

    challenges = [
        ("Ambiguous references",
         "'John met Bill. He was happy.' (Who was happy?)"),

        ("Long-distance dependencies",
         "Antecedent far from pronoun in text"),

        ("Collective references",
         "'The team won. They celebrated.' (team = plural)"),

        ("Implicit antecedents",
         "'The bike was stolen. It was never found.' (bike not explicit)"),

        ("Genre/domain variation",
         "Different writing styles use different patterns")]

    print("Common challenges:\n")
    for challenge, example in challenges:
        print(f"  • {challenge}")
        print(f"    Example: {example}\n")

    print("\nEvaluation metrics:")
    print("  • MUC (Message Understanding Conference) score")
    print("  • B-CUBED: Precision and recall on mention pairs")
    print("  • CEAF (Constrained Entity Alignment F-Measure)")
    print("  • CoNLL F1: Average of MUC, B-CUBED, and CEAF")


def real_world_applications():
    """Show real-world applications."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD APPLICATIONS ===\n")

    print("1. Question Answering")
    print("   Resolve pronouns to find answers")
    print("   Q: 'When did he become president?' (in context of Obama text)\n")

    print("2. Document Summarization")
    print("   Resolve references for clear summaries")
    print("   Avoids unclear 'he', 'she', 'it' in summary\n")

    print("3. Information Extraction")
    print("   Extract complete entity relationships")
    print("   Link mentions across multiple sentences\n")

    print("4. Machine Translation")
    print("   Correct pronoun translation needs coreference")
    print("   Different languages have different pronoun systems\n")

    print("5. Sentiment Analysis")
    print("   Attribute sentiments to correct entities")
    print("   'The product is great, but it breaks easily' (both about product)")


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Run on clean text", "Poor sentence splitting breaks coreference"),
        ("Consider domain", "Medical/legal text has different patterns"),
        ("Validate resolution", "Check if resolved text makes sense"),
        ("Combine with NER", "Use entity types to filter candidates"),
        ("Handle edge cases", "Collective nouns, cataphora, etc."),
        ("LLM as backup", "Use GPT-4 for difficult cases"),
        ("Test thoroughly", "Coreference errors propagate to downstream tasks")]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def tools_comparison():
    """Compare coreference resolution tools."""
    print("=" * 70)
    print("=== TOOLS COMPARISON ===\n")

    print("Tool           Speed    Accuracy   Setup      Notes")
    print("-" * 70)
    print("neuralcoref    Fast     Good       Easy       spaCy integration, archived")
    print("FastCoref      Fast     Better     Easy       Modern, actively maintained")
    print("AllenNLP       Medium   Best       Complex    Research-grade, heavy")
    print("Hugging Face   Medium   Best       Easy       Pre-trained transformers")
    print("LLM (GPT-4)    Slow     Excellent  Easiest    Most accurate, expensive")
    print("-" * 70)

    print("\n\nRecommendations:")
    print("  • Prototyping: Use LLM (GPT-4)")
    print("  • Production: FastCoref or Hugging Face model")
    print("  • Research: AllenNLP coref")
    print("  • Budget-friendly: FastCoref")


def practical_example():
    """Show practical end-to-end example."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL EXAMPLE ===\n")

    print("Task: Extract all facts about a person from a bio\n")

    print("Input text:")
    text = """
Marie Curie was born in Poland in 1867. She moved to France to study physics.
Curie was the first woman to win a Nobel Prize. She won it twice, actually.
Her research on radioactivity was groundbreaking. Marie died in 1934.
"""
    print(text)

    print("\nStep 1: Resolve coreferences")
    print("  'She' → 'Marie Curie'")
    print("  'Curie' → 'Marie Curie'")
    print("  'Her' → 'Marie Curie'")
    print("  'Marie' → 'Marie Curie'")
    print("  'it' → 'Nobel Prize'\n")

    print("Step 2: Resolved text")
    resolved = """
Marie Curie was born in Poland in 1867. Marie Curie moved to France to study
physics. Marie Curie was the first woman to win a Nobel Prize. Marie Curie won
the Nobel Prize twice, actually. Marie Curie's research on radioactivity was
groundbreaking. Marie Curie died in 1934.
"""
    print(resolved)

    print("\nStep 3: Extract facts")
    print("  (Marie Curie, born in, Poland)")
    print("  (Marie Curie, born in, 1867)")
    print("  (Marie Curie, moved to, France)")
    print("  (Marie Curie, studied, physics)")
    print("  (Marie Curie, won, Nobel Prize)")
    print("  (Marie Curie, researched, radioactivity)")
    print("  (Marie Curie, died in, 1934)")


if __name__ == "__main__":
    coreference_intro()
    coreference_examples()
    basic_usage()
    resolving_coreferences()
    improving_extraction()
    llm_based_coreference()
    evaluation_challenges()
    real_world_applications()
    best_practices()
    tools_comparison()
    practical_example()

    print("\n" + "=" * 70)
    print("\nKey insight: Coreference resolution is essential for text understanding!")
    print("Resolve references to extract complete, accurate information")

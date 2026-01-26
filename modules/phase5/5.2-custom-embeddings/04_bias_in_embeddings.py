"""
04 - Bias in Embeddings
========================
Understand and mitigate biases in embedding models.

Key concept: Embeddings can encode societal biases from training data. Being aware of these biases is crucial for building fair AI systems.

Book reference: speach_lang.I.6.11
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import numpy as np
from typing import List, Dict, Tuple


def bias_in_embeddings_intro():
    """Introduce bias in embeddings."""
    print("=== BIAS IN EMBEDDINGS ===\n")

    print("What is embedding bias?")
    print("  Systematic associations between concepts that reflect societal stereotypes")
    print("  or historical inequities present in training data.\n")

    print("Example biases:")
    print("  • Gender: 'doctor' closer to 'man', 'nurse' closer to 'woman'")
    print("  • Race: Names associated with different racial groups treated differently")
    print("  • Age: 'young' associated with 'technology', 'old' with 'slow'")
    print("  • Socioeconomic: Certain occupations linked to class stereotypes\n")

    print("Why it matters:")
    print("  • Biased embeddings lead to biased AI systems")
    print("  • Can perpetuate discrimination in hiring, lending, healthcare")
    print("  • Legal and ethical implications")
    print("  • Damage to user trust and brand reputation\n")

    print("Sources of bias:")
    print("  • Training data reflects historical biases")
    print("  • Sampling bias in data collection")
    print("  • Amplification of existing stereotypes")
    print("  • Lack of diverse representation")


def types_of_bias():
    """Explain different types of bias."""
    print("\n" + "=" * 70)
    print("=== TYPES OF BIAS ===\n")

    print("1. Gender Bias")
    print("   Associations between gender and occupations, traits, activities")
    print("   Example: 'programmer' more similar to 'he' than 'she'\n")

    print("2. Racial Bias")
    print("   Different associations for names/words related to different races")
    print("   Example: African American names rated more negatively\n")

    print("3. Age Bias")
    print("   Stereotypical associations with age groups")
    print("   Example: 'innovative' closer to 'young' than 'experienced'\n")

    print("4. Socioeconomic Bias")
    print("   Class-based stereotypes in word associations")
    print("   Example: 'wealthy' associated with 'intelligent'\n")

    print("5. Religious Bias")
    print("   Prejudicial associations with religious groups")
    print("   Example: Certain religions associated with violence\n")

    print("6. Disability Bias")
    print("   Negative associations with disability-related terms")
    print("   Example: 'disabled' closer to 'unable' than 'capable'")


def measuring_bias():
    """Show how to measure bias in embeddings."""
    print("\n" + "=" * 70)
    print("=== MEASURING BIAS ===\n")

    print("Method 1: Word Embedding Association Test (WEAT)")
    print("  Measures differential association between two sets of target words")
    print("  and two sets of attribute words.\n")

    print("Example - Gender bias in occupations:")
    print("  Target set A: ['programmer', 'engineer', 'scientist']")
    print("  Target set B: ['nurse', 'teacher', 'librarian']")
    print("  Attribute set X: ['he', 'him', 'his', 'man', 'male']")
    print("  Attribute set Y: ['she', 'her', 'hers', 'woman', 'female']\n")

    print("  Calculate: How much more is A associated with X than Y?")
    print("  Effect size d > 0 indicates bias\n")

    code_weat = '''def weat_score(model, target_a, target_b, attr_x, attr_y):
    """
    Calculate WEAT score for measuring bias.

    Returns effect size d (Cohen's d)
    d > 0.8: Large bias
    d = 0.5-0.8: Medium bias
    d < 0.5: Small bias
    """
    # Get embeddings
    emb_a = [model.encode(word) for word in target_a]
    emb_b = [model.encode(word) for word in target_b]
    emb_x = [model.encode(word) for word in attr_x]
    emb_y = [model.encode(word) for word in attr_y]

    def association(target, attr_x, attr_y):
        # Mean cosine similarity with X minus mean with Y
        sim_x = np.mean([cosine_similarity(target, x) for x in attr_x])
        sim_y = np.mean([cosine_similarity(target, y) for y in attr_y])
        return sim_x - sim_y

    # Calculate associations
    s_a = [association(a, emb_x, emb_y) for a in emb_a]
    s_b = [association(b, emb_x, emb_y) for b in emb_b]

    # Effect size (Cohen's d)
    mean_diff = np.mean(s_a) - np.mean(s_b)
    pooled_std = np.sqrt((np.std(s_a)**2 + np.std(s_b)**2) / 2)
    effect_size = mean_diff / pooled_std

    return effect_size

# Example usage
target_a = ['programmer', 'engineer', 'scientist']
target_b = ['nurse', 'teacher', 'librarian']
attr_male = ['he', 'him', 'his', 'man', 'male']
attr_female = ['she', 'her', 'hers', 'woman', 'female']

bias_score = weat_score(model, target_a, target_b, attr_male, attr_female)
print(f"Gender-occupation bias: {bias_score:.3f}")
'''

    print(code_weat)

    print("\n\nMethod 2: Projection onto bias direction")
    print("  Project embeddings onto gender direction (he-she)")
    print("  Measure how biased each word is along this axis\n")

    code_proj = '''def bias_projection(model, word, direction_a, direction_b):
    """
    Project word onto bias direction (e.g., he-she).

    Returns: Projection magnitude (-1 to 1)
    Positive = biased toward direction_a
    Negative = biased toward direction_b
    """
    # Get embeddings
    emb_word = model.encode(word)
    emb_a = model.encode(direction_a)  # e.g., 'he'
    emb_b = model.encode(direction_b)  # e.g., 'she'

    # Bias direction
    bias_dir = emb_a - emb_b
    bias_dir = bias_dir / np.linalg.norm(bias_dir)

    # Project word onto bias direction
    projection = np.dot(emb_word, bias_dir)

    return projection

# Test occupations
occupations = ['doctor', 'nurse', 'engineer', 'teacher', 'CEO', 'secretary']
for occ in occupations:
    proj = bias_projection(model, occ, 'he', 'she')
    gender = "male" if proj > 0 else "female"
    print(f"{occ:12} {proj:+.3f} (bias toward {gender})")

# Example output:
# doctor       +0.245 (bias toward male)
# nurse        -0.312 (bias toward female)
# engineer     +0.189 (bias toward male)
'''

    print(code_proj)


def bias_examples():
    """Show concrete bias examples."""
    print("\n" + "=" * 70)
    print("=== BIAS EXAMPLES ===\n")

    print("Example 1: Gender bias in occupations\n")

    print("Male-biased occupations:")
    print("  • programmer, engineer, architect, CEO")
    print("  • Embeddings closer to 'he/man' than 'she/woman'\n")

    print("Female-biased occupations:")
    print("  • nurse, teacher, secretary, receptionist")
    print("  • Embeddings closer to 'she/woman' than 'he/man'\n")

    print("Impact:")
    print("  → Resume screening AI may rank men higher for tech roles")
    print("  → Search results may show stereotypical gender images\n")

    print("\n" + "-" * 70)
    print("\nExample 2: Name-based racial bias\n")

    print("Study finding:")
    print("  Resumes with 'white-sounding' names (Emily, Greg)")
    print("  vs 'Black-sounding' names (Lakisha, Jamal)")
    print("  → Same qualifications, different response rates\n")

    print("In embeddings:")
    print("  African American names may be closer to negative attributes")
    print("  European American names closer to positive attributes\n")

    print("Impact:")
    print("  → Biased chatbots, hiring tools, credit scoring\n")

    print("\n" + "-" * 70)
    print("\nExample 3: Sentiment bias\n")

    print("Sentence: 'They are [adjective]'")
    print("  When subject is African American → more negative")
    print("  When subject is European American → more positive\n")

    print("Impact:")
    print("  → Sentiment analysis may rate same text differently by race")
    print("  → Content moderation may be harsher on certain groups")


def mitigating_bias():
    """Explain bias mitigation techniques."""
    print("\n" + "=" * 70)
    print("=== MITIGATING BIAS ===\n")

    print("1. Data-level interventions")
    print("   • Collect more balanced training data")
    print("   • Remove biased examples from training set")
    print("   • Data augmentation to balance representations")
    print("   ✓ Addresses root cause")
    print("   ✗ Requires retraining, may be expensive\n")

    print("2. Post-processing debiasing")
    print("   • Remove bias direction from embeddings")
    print("   • Project out gender/race dimensions")
    print("   • Equalize embeddings for protected groups")
    print("   ✓ Can be applied to existing embeddings")
    print("   ✗ May degrade embedding quality")
    print("   ✗ Hard to remove all biases\n")

    print("3. Adversarial debiasing")
    print("   • Train model to not encode protected attributes")
    print("   • Use adversarial loss during training")
    print("   ✓ Effective at removing specific biases")
    print("   ✗ Requires retraining with specialized setup\n")

    print("4. Counterfactual data augmentation")
    print("   • Generate examples with swapped protected attributes")
    print("   • Train on original + counterfactual examples")
    print("   Example: 'He is a nurse' → 'She is a nurse'")
    print("   ✓ Reduces specific biases")
    print("   ✗ May not generalize to all biases")


def debiasing_code_example():
    """Show debiasing code example."""
    print("\n" + "=" * 70)
    print("=== DEBIASING CODE EXAMPLE ===\n")

    print("Hard debiasing algorithm (Bolukbasi et al., 2016):\n")

    code = '''def debias_embeddings(embeddings, word_to_idx, bias_pairs):
    """
    Remove bias direction from embeddings.

    bias_pairs: [(word1, word2), ...] defining bias direction
                e.g., [('he', 'she'), ('him', 'her')]
    """
    # 1. Identify bias direction
    # Average difference vectors for bias-defining pairs
    bias_vectors = []
    for word1, word2 in bias_pairs:
        idx1 = word_to_idx[word1]
        idx2 = word_to_idx[word2]
        diff = embeddings[idx1] - embeddings[idx2]
        bias_vectors.append(diff)

    # Bias direction = first principal component
    bias_direction = np.mean(bias_vectors, axis=0)
    bias_direction = bias_direction / np.linalg.norm(bias_direction)

    # 2. Remove bias component from all embeddings
    debiased = embeddings.copy()
    for i in range(len(embeddings)):
        # Project onto bias direction
        projection = np.dot(embeddings[i], bias_direction)

        # Remove bias component
        debiased[i] = embeddings[i] - projection * bias_direction

    return debiased

# Example usage
bias_pairs = [
    ('he', 'she'),
    ('him', 'her'),
    ('his', 'hers'),
    ('man', 'woman'),
    ('male', 'female')
]

debiased_embeddings = debias_embeddings(
    original_embeddings,
    word_to_idx,
    bias_pairs
)

# Verify: Check gender bias before/after
print("Before debiasing:")
print(f"  doctor-he similarity: {cosine_sim('doctor', 'he', original):.3f}")
print(f"  doctor-she similarity: {cosine_sim('doctor', 'she', original):.3f}")

print("\\nAfter debiasing:")
print(f"  doctor-he similarity: {cosine_sim('doctor', 'he', debiased):.3f}")
print(f"  doctor-she similarity: {cosine_sim('doctor', 'she', debiased):.3f}")
'''

    print(code)


def limitations_and_challenges():
    """Discuss limitations of bias mitigation."""
    print("\n" + "=" * 70)
    print("=== LIMITATIONS AND CHALLENGES ===\n")

    print("Challenges:\n")

    challenges = [
        ("Multiple biases", "Can't remove all biases simultaneously"),
        ("Definition of fairness", "No consensus on what 'fair' means"),
        ("Performance trade-offs", "Debiasing may reduce embedding quality"),
        ("Incomplete mitigation", "Biases may manifest in subtle ways"),
        ("New biases", "Removing one bias may introduce others"),
        ("Evaluation difficulty", "Hard to measure all possible biases"),
    ]

    for challenge, description in challenges:
        print(f"  • {challenge}: {description}")

    print("\n\nImportant considerations:")
    print("  ⚠ Debiasing is not a complete solution")
    print("  ⚠ May create illusion of fairness while bias remains")
    print("  ⚠ Cannot replace careful system design and auditing")
    print("  ⚠ Different applications may need different fairness criteria")


def responsible_ai_practices():
    """List responsible AI practices for embeddings."""
    print("\n" + "=" * 70)
    print("=== RESPONSIBLE AI PRACTICES ===\n")

    practices = [
        ("Measure bias regularly", "Test for known biases in your domain"),
        ("Diverse training data", "Ensure representation across groups"),
        ("Document limitations", "Be transparent about known biases"),
        ("Monitor in production", "Track fairness metrics over time"),
        ("Human oversight", "Don't fully automate sensitive decisions"),
        ("Provide explanations", "Help users understand system behavior"),
        ("Allow appeals", "Let users challenge unfair outcomes"),
        ("Regular audits", "Third-party fairness assessments"),
        ("Diverse team", "Include diverse perspectives in development"),
        ("Consider context", "Understand how biases affect your use case"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def bias_testing_checklist():
    """Provide bias testing checklist."""
    print("=" * 70)
    print("=== BIAS TESTING CHECKLIST ===\n")

    checklist = [
        "□ Test for gender bias in occupation associations",
        "□ Test for racial bias in name associations",
        "□ Test for age bias in capability associations",
        "□ Test for disability bias in sentiment",
        "□ Test for religious bias in threat associations",
        "□ Measure WEAT scores for relevant dimensions",
        "□ Analyze bias projections for key terms",
        "□ Test on domain-specific sensitive terms",
        "□ Compare bias across different demographic groups",
        "□ Document all findings",
        "□ Consider legal and ethical implications",
        "□ Plan mitigation strategies",
    ]

    for item in checklist:
        print(f"  {item}")


def when_to_debias():
    """Guide on when to apply debiasing."""
    print("\n" + "=" * 70)
    print("=== WHEN TO DEBIAS? ===\n")

    print("Debias when:\n")

    print("  • High-stakes decisions (hiring, lending, healthcare)")
    print("  • Legal requirements (anti-discrimination laws)")
    print("  • Measured bias is significant (WEAT d > 0.5)")
    print("  • Public-facing systems")
    print("  • Vulnerable populations involved\n")

    print("Consider not debiasing when:\n")

    print("  • Bias is minimal (WEAT d < 0.3)")
    print("  • Low-stakes application")
    print("  • Performance degradation is unacceptable")
    print("  • Internal tool with human oversight\n")

    print("Always:")
    print("  • Document bias levels")
    print("  • Implement monitoring")
    print("  • Provide transparency to users")
    print("  • Plan for regular audits")


def further_resources():
    """List resources for learning more."""
    print("\n" + "=" * 70)
    print("=== FURTHER RESOURCES ===\n")

    print("Key papers:")
    print("  • Bolukbasi et al. (2016) - 'Man is to Computer Programmer as")
    print("    Woman is to Homemaker? Debiasing Word Embeddings'")
    print("  • Caliskan et al. (2017) - 'Semantics derived automatically from")
    print("    language corpora contain human-like biases'")
    print("  • Sun et al. (2019) - 'Mitigating Gender Bias in Natural Language")
    print("    Processing: Literature Review'\n")

    print("Tools:")
    print("  • AI Fairness 360 (IBM): Bias detection and mitigation")
    print("  • Fairlearn (Microsoft): Fairness assessment")
    print("  • What-If Tool (Google): Visual bias analysis\n")

    print("Guidelines:")
    print("  • EU AI Act: Regulations on high-risk AI systems")
    print("  • NIST AI Risk Management Framework")
    print("  • ACM Code of Ethics: Computing professionals guidelines")


if __name__ == "__main__":
    bias_in_embeddings_intro()
    types_of_bias()
    measuring_bias()
    bias_examples()
    mitigating_bias()
    debiasing_code_example()
    limitations_and_challenges()
    responsible_ai_practices()
    bias_testing_checklist()
    when_to_debias()
    further_resources()

    print("\n" + "=" * 70)
    print("\nKey insight: Embeddings encode societal biases from training data")
    print("Awareness, measurement, and mitigation are essential for responsible AI!")

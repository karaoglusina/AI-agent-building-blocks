"""
03 - Generation Metrics
=======================
BLEU, ROUGE for text generation evaluation.

Key concept: These metrics measure overlap with reference text - useful but limited.

Book reference: AI_eng.3, speach_lang.II.13.6
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def calculate_bleu(reference: str, candidate: str) -> dict:
    """Calculate BLEU score between reference and candidate."""
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    
    # Use smoothing to handle zero n-gram matches
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU at different n-gram levels
    bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_4": bleu_4}


def calculate_rouge(reference: str, candidate: str) -> dict:
    """Calculate ROUGE scores between reference and candidate."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def evaluate_generation(reference: str, candidate: str) -> dict:
    """Calculate all generation metrics."""
    bleu = calculate_bleu(reference, candidate)
    rouge = calculate_rouge(reference, candidate)
    return {**bleu, **rouge}


# Example: summarization evaluation
REFERENCE_SUMMARIES = [
    "Senior Python developer position with Django and PostgreSQL experience required.",
    "The role requires 5 years of experience and offers remote work options."]

CANDIDATE_SUMMARIES = [
    # Good summary
    "Senior Python developer role requiring Django and PostgreSQL skills.",
    # Okay summary
    "Looking for a Python developer with database experience.",
    # Poor summary  
    "This is a software engineering position at a tech company."]


if __name__ == "__main__":
    print("=== GENERATION METRICS ===\n")
    
    reference = REFERENCE_SUMMARIES[0]
    print(f"Reference: \"{reference}\"\n")
    
    for i, candidate in enumerate(CANDIDATE_SUMMARIES, 1):
        print(f"--- Candidate {i} ---")
        print(f"\"{candidate}\"")
        
        metrics = evaluate_generation(reference, candidate)
        
        print(f"\nBLEU-1: {metrics['bleu_1']:.3f} (unigram overlap)")
        print(f"BLEU-4: {metrics['bleu_4']:.3f} (n-gram up to 4)")
        print(f"ROUGE-1: {metrics['rouge1_f']:.3f} (unigram F1)")
        print(f"ROUGE-L: {metrics['rougeL_f']:.3f} (longest common subsequence)")
        print()
    
    print("=== METRIC INTERPRETATION ===")
    print("BLEU: Measures n-gram precision (how much of candidate is in reference)")
    print("ROUGE: Measures recall (how much of reference is in candidate)")
    print()
    print("Limitations:")
    print("• Can't capture semantic similarity")
    print("• Penalize valid paraphrases")
    print("• Don't measure fluency or coherence")
    print("• Best used alongside human evaluation or LLM-as-judge")

"""
06 - Fine-tune Evaluation
==========================
Measure improvement after fine-tuning.

Key concept: Proper evaluation is critical to determine if fine-tuning improved your model. Use both automatic metrics and human evaluation.

Book reference: AI_eng.7, hands_on_LLM.III.12
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from typing import List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class EvaluationResult:
    """Holds evaluation metrics."""
    metric_name: str
    base_model_score: float
    finetuned_score: float

    def improvement(self) -> float:
        """Calculate improvement percentage."""
        return ((self.finetuned_score - self.base_model_score) / self.base_model_score) * 100

    def __str__(self) -> str:
        return (f"{self.metric_name}: {self.base_model_score:.3f} → {self.finetuned_score:.3f} "
                f"({self.improvement():+.1f}%)")


def evaluation_types():
    """Explain different types of evaluation."""
    print("=== TYPES OF EVALUATION ===\n")

    print("1. Automatic Metrics")
    print("   Computed by algorithms, fast and scalable")
    print("   Examples: Perplexity, BLEU, ROUGE, Exact Match")
    print("   ✓ Fast, cheap, reproducible")
    print("   ✗ May not correlate with human judgment\n")

    print("2. Human Evaluation")
    print("   Manual review by humans")
    print("   Examples: Quality ratings, preference ranking")
    print("   ✓ Reflects true quality")
    print("   ✗ Slow, expensive, subjective\n")

    print("3. LLM-as-Judge")
    print("   Use strong LLM (GPT-4) to evaluate outputs")
    print("   Examples: Quality scoring, preference comparison")
    print("   ✓ Faster and cheaper than humans")
    print("   ✗ May have biases, not always reliable\n")

    print("4. Task-Specific Metrics")
    print("   Metrics specific to your task")
    print("   Examples: Accuracy (classification), F1 (extraction)")
    print("   ✓ Directly measures task performance")
    print("   ✓ Easy to interpret")


def perplexity_explained():
    """Explain perplexity as evaluation metric."""
    print("\n" + "=" * 70)
    print("=== PERPLEXITY ===\n")

    print("What is perplexity?")
    print("  A measure of how 'surprised' the model is by test data.")
    print("  Lower perplexity = better language modeling.\n")

    print("Formula:")
    print("  Perplexity = exp(average cross-entropy loss)")
    print("  PPL = exp(-(1/N) × Σ log P(word_i | context))\n")

    print("Interpretation:")
    print("  PPL = 10  → Model is choosing from ~10 words on average (good)")
    print("  PPL = 100 → Model is choosing from ~100 words (worse)")
    print("  PPL = 1   → Perfect prediction (unrealistic)\n")

    print("Example:")
    # Simple perplexity calculation
    losses = [0.5, 0.7, 0.4, 0.6]
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)

    print(f"  Average loss: {avg_loss:.3f}")
    print(f"  Perplexity: {perplexity:.3f}\n")

    print("When to use:")
    print("  ✓ General language modeling tasks")
    print("  ✓ Comparing model checkpoints")
    print("  ✓ Quick quality check")
    print("  ✗ Doesn't measure task-specific performance")
    print("  ✗ Can be misleading for generation quality")


def classification_metrics():
    """Explain metrics for classification tasks."""
    print("\n" + "=" * 70)
    print("=== CLASSIFICATION METRICS ===\n")

    print("For multi-class classification tasks:\n")

    print("Accuracy:")
    print("  Percentage of correct predictions")
    print("  Formula: Correct / Total")
    print("  Example: 85/100 = 85% accuracy\n")

    print("Precision:")
    print("  Of predicted positives, how many are correct?")
    print("  Formula: True Positives / (True Positives + False Positives)")
    print("  Example: Spam detection - 90 real spam / 100 marked spam = 90%\n")

    print("Recall:")
    print("  Of actual positives, how many did we find?")
    print("  Formula: True Positives / (True Positives + False Negatives)")
    print("  Example: Found 90 spam / 120 total spam = 75%\n")

    print("F1 Score:")
    print("  Harmonic mean of precision and recall")
    print("  Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print("  Example: 2 × (0.90 × 0.75) / (0.90 + 0.75) = 0.82\n")

    print("Example comparison:")
    print("  Metric      Base Model    Fine-tuned    Improvement")
    print("  " + "-" * 60)
    print("  Accuracy    65%           82%           +17%")
    print("  Precision   62%           85%           +23%")
    print("  Recall      70%           80%           +10%")
    print("  F1 Score    66%           82%           +16%")


def generation_metrics():
    """Explain metrics for text generation."""
    print("\n" + "=" * 70)
    print("=== GENERATION METRICS ===\n")

    print("BLEU (Bilingual Evaluation Understudy)")
    print("  Measures n-gram overlap with reference")
    print("  Range: 0-1 (higher is better)")
    print("  Use: Translation, summarization")
    print("  Example: BLEU-4 = 0.35 (moderate overlap)\n")

    print("ROUGE (Recall-Oriented Understudy for Gisting Evaluation)")
    print("  Measures recall of n-grams")
    print("  Variants: ROUGE-1, ROUGE-2, ROUGE-L")
    print("  Use: Summarization")
    print("  Example: ROUGE-L = 0.45 (good summary)\n")

    print("Exact Match")
    print("  Percentage of exactly correct outputs")
    print("  Range: 0-100%")
    print("  Use: QA, structured generation")
    print("  Example: 73% exact match on validation set\n")

    print("Limitations:")
    print("  ✗ Don't capture semantic similarity well")
    print("  ✗ Multiple correct answers penalized")
    print("  ✗ Don't measure fluency or coherence")
    print("  ✗ Can be gamed by memorization")


def llm_as_judge():
    """Explain using LLM as evaluation judge."""
    print("\n" + "=" * 70)
    print("=== LLM-AS-JUDGE EVALUATION ===\n")

    print("Concept:")
    print("  Use a strong LLM (GPT-4, Claude) to evaluate model outputs\n")

    print("Evaluation prompts:\n")

    print("Quality scoring:")
    print('''  "Rate the quality of this response on a scale of 1-5:
  Prompt: {prompt}
  Response: {response}

  Consider: Accuracy, helpfulness, clarity
  Rating (1-5):"''')

    print("\n\nPreference comparison:")
    print('''  "Which response is better for this prompt?
  Prompt: {prompt}

  Response A: {base_model_response}
  Response B: {finetuned_response}

  Choose: A or B"''')

    print("\n\nBenefits:")
    print("  ✓ Captures semantic quality")
    print("  ✓ Faster than human evaluation")
    print("  ✓ Can evaluate many dimensions (helpfulness, safety, etc.)")
    print("  ✓ Scales well\n")

    print("Limitations:")
    print("  ✗ Costs money (API calls)")
    print("  ✗ May prefer certain styles")
    print("  ✗ Can be inconsistent")
    print("  ✗ Strong model may not be available")


def evaluation_protocol():
    """Show proper evaluation protocol."""
    print("\n" + "=" * 70)
    print("=== EVALUATION PROTOCOL ===\n")

    print("Step 1: Prepare test set")
    print("  • Hold out 10-20% of data for testing")
    print("  • NEVER train on test data")
    print("  • Ensure test set represents real use cases")
    print("  • Stratify by category if applicable\n")

    print("Step 2: Baseline evaluation")
    print("  • Evaluate base model on test set")
    print("  • Record all metrics")
    print("  • Save outputs for comparison\n")

    print("Step 3: Fine-tuned evaluation")
    print("  • Evaluate fine-tuned model on SAME test set")
    print("  • Use SAME metrics")
    print("  • Save outputs for comparison\n")

    print("Step 4: Compare results")
    print("  • Calculate improvement per metric")
    print("  • Statistical significance testing")
    print("  • Qualitative analysis of outputs\n")

    print("Step 5: Error analysis")
    print("  • Review cases where fine-tuned model fails")
    print("  • Review cases where fine-tuned model improves")
    print("  • Identify patterns and areas for improvement")


def evaluation_checklist():
    """Provide evaluation checklist."""
    print("\n" + "=" * 70)
    print("=== EVALUATION CHECKLIST ===\n")

    checklist = [
        "□ Test set held out (never trained on)",
        "□ Test set size sufficient (100+ examples)",
        "□ Test set representative of real use",
        "□ Baseline model evaluated on test set",
        "□ Fine-tuned model evaluated on test set",
        "□ Same evaluation protocol for both",
        "□ Multiple metrics computed",
        "□ Statistical significance tested",
        "□ Qualitative analysis done",
        "□ Error cases reviewed",
        "□ Improvement documented",
        "□ Outputs saved for later review",
    ]

    for item in checklist:
        print(f"  {item}")


def common_pitfalls():
    """Discuss common evaluation pitfalls."""
    print("\n" + "=" * 70)
    print("=== COMMON PITFALLS ===\n")

    pitfalls = [
        ("Train/test leakage",
         "Accidentally training on test data → inflated scores"),

        ("Cherry-picking metrics",
         "Only reporting metrics that look good → misleading"),

        ("Small test set",
         "Test set too small → unreliable results"),

        ("Single metric focus",
         "Only looking at accuracy → missing other issues"),

        ("Ignoring qualitative analysis",
         "Not reviewing actual outputs → missing critical failures"),

        ("No baseline comparison",
         "Not comparing to base model → can't measure improvement"),

        ("Overfitting to validation set",
         "Tuning on validation set → need separate test set"),

        ("Ignoring edge cases",
         "Only testing common cases → failures in production"),
    ]

    for pitfall, consequence in pitfalls:
        print(f"✗ {pitfall}")
        print(f"  → {consequence}\n")


def evaluation_example():
    """Show comprehensive evaluation example."""
    print("=" * 70)
    print("=== EVALUATION EXAMPLE ===\n")

    print("Task: Customer support classification (5 categories)\n")

    print("Results:\n")

    results = [
        EvaluationResult("Accuracy", 68.5, 87.2),
        EvaluationResult("F1 Score", 65.3, 85.8),
        EvaluationResult("Perplexity", 12.5, 8.3),
    ]

    for result in results:
        print(f"  {result}")

    print("\n\nQualitative findings:")
    print("  ✓ Better at handling ambiguous queries")
    print("  ✓ More consistent formatting")
    print("  ✓ Fewer hallucinations")
    print("  ✗ Still struggles with rare categories")
    print("  ✗ Occasionally too verbose\n")

    print("Conclusion:")
    print("  Fine-tuning significantly improved performance (+27% accuracy)")
    print("  Model ready for production with monitoring for rare cases")


def continuous_evaluation():
    """Discuss continuous evaluation in production."""
    print("\n" + "=" * 70)
    print("=== CONTINUOUS EVALUATION ===\n")

    print("Monitor in production:")
    print("  • Track metrics over time")
    print("  • A/B test base vs fine-tuned model")
    print("  • Collect user feedback (thumbs up/down)")
    print("  • Monitor edge cases and failures")
    print("  • Detect distribution drift\n")

    print("Red flags:")
    print("  ⚠ Metrics degrading over time")
    print("  ⚠ Increased error rate")
    print("  ⚠ More user complaints")
    print("  ⚠ New types of failures appearing\n")

    print("Actions:")
    print("  → Collect new training data")
    print("  → Retrain with updated data")
    print("  → Adjust hyperparameters")
    print("  → Consider switching base model")


def best_practices():
    """List evaluation best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Use multiple metrics - No single metric tells the full story",
        "2. Include human evaluation - Especially for generation tasks",
        "3. Test on diverse inputs - Cover all use cases and edge cases",
        "4. Compare to baseline - Always measure improvement",
        "5. Do error analysis - Understand where model fails",
        "6. Test worst cases - Push model to its limits",
        "7. Measure latency - Speed matters in production",
        "8. Check consistency - Same input should give similar outputs",
        "9. Validate safety - Test for harmful outputs",
        "10. Document everything - Track metrics, configs, findings",
    ]

    for practice in practices:
        print(f"  {practice}")


if __name__ == "__main__":
    evaluation_types()
    perplexity_explained()
    classification_metrics()
    generation_metrics()
    llm_as_judge()
    evaluation_protocol()
    evaluation_checklist()
    common_pitfalls()
    evaluation_example()
    continuous_evaluation()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Rigorous evaluation is essential!")
    print("Measure multiple metrics, do error analysis, and monitor in production")

"""
05 - SFT/RLHF/DPO Overview
===========================
Understanding alignment techniques for LLMs.

Key concept: After pre-training, LLMs are aligned with human preferences through Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), or Direct Preference Optimization (DPO).

Book reference: AI_eng.7, hands_on_LLM.III.12, speach_lang.I.12.7
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def llm_training_pipeline():
    """Explain the full LLM training pipeline."""
    print("=== LLM TRAINING PIPELINE ===\n")

    print("Modern LLMs go through multiple training stages:\n")

    print("Stage 1: Pre-training")
    print("  • Train on massive text corpus (trillions of tokens)")
    print("  • Learn language patterns, world knowledge")
    print("  • Objective: Predict next token")
    print("  • Result: Base model (e.g., Llama-2-base)\n")

    print("Stage 2: Supervised Fine-Tuning (SFT)")
    print("  • Train on instruction-following examples")
    print("  • Format: (instruction, response) pairs")
    print("  • Objective: Learn to follow instructions")
    print("  • Result: Instruction-tuned model (e.g., Llama-2-7b-chat)\n")

    print("Stage 3: Alignment (RLHF or DPO)")
    print("  • Align with human preferences")
    print("  • Format: Preferred vs rejected responses")
    print("  • Objective: Maximize human preference")
    print("  • Result: Aligned model (safe, helpful, harmless)\n")

    print("Example model names:")
    print("  llama-2-7b          → After pre-training")
    print("  llama-2-7b-chat     → After SFT + RLHF")
    print("  mistral-7b-v0.1     → After pre-training")
    print("  mistral-7b-instruct → After SFT + alignment")


def supervised_fine_tuning():
    """Explain Supervised Fine-Tuning (SFT)."""
    print("\n" + "=" * 70)
    print("=== SUPERVISED FINE-TUNING (SFT) ===\n")

    print("What is SFT?")
    print("  Fine-tuning a pre-trained model on (instruction, response) pairs")
    print("  to teach it to follow instructions.\n")

    print("Training data format:")
    print("  Input:    'Summarize this article: [article text]'")
    print("  Output:   '[summary]'\n")

    print("  Input:    'Translate to French: Hello, how are you?'")
    print("  Output:   'Bonjour, comment allez-vous?'\n")

    print("Key characteristics:")
    print("  ✓ Standard supervised learning")
    print("  ✓ Straightforward to implement")
    print("  ✓ Requires high-quality instruction-response pairs")
    print("  ✓ Typically 10K-100K examples")
    print("  ✗ Doesn't optimize for human preferences directly")
    print("  ✗ May not capture nuanced preferences\n")

    print("When to use:")
    print("  • Teaching specific instruction-following behavior")
    print("  • Domain adaptation with instructions")
    print("  • First step before RLHF/DPO")
    print("  • When you have clear input-output pairs")


def rlhf_explained():
    """Explain Reinforcement Learning from Human Feedback (RLHF)."""
    print("\n" + "=" * 70)
    print("=== RLHF (REINFORCEMENT LEARNING FROM HUMAN FEEDBACK) ===\n")

    print("What is RLHF?")
    print("  Using human preference data to train a reward model,")
    print("  then optimizing the LLM with reinforcement learning.\n")

    print("RLHF Pipeline:\n")

    print("Step 1: Collect preference data")
    print("  • Generate multiple responses for same prompt")
    print("  • Humans rank responses (best to worst)")
    print("  • Result: Preference dataset\n")

    print("Step 2: Train reward model")
    print("  • Train classifier to predict human preference")
    print("  • Input: (prompt, response)")
    print("  • Output: Score (higher = better)")
    print("  • Result: Reward model\n")

    print("Step 3: RL optimization (PPO)")
    print("  • Use reward model to guide LLM training")
    print("  • Algorithm: PPO (Proximal Policy Optimization)")
    print("  • Maximize reward while staying close to base model")
    print("  • Result: Aligned model\n")

    print("Benefits:")
    print("  ✓ Aligns with true human preferences")
    print("  ✓ Can capture nuanced quality differences")
    print("  ✓ Used by OpenAI (GPT-4), Anthropic (Claude)")
    print("  ✗ Complex to implement")
    print("  ✗ Expensive (requires many comparisons)")
    print("  ✗ Can be unstable during training")


def dpo_explained():
    """Explain Direct Preference Optimization (DPO)."""
    print("\n" + "=" * 70)
    print("=== DPO (DIRECT PREFERENCE OPTIMIZATION) ===\n")

    print("What is DPO?")
    print("  A simpler alternative to RLHF that directly optimizes")
    print("  the LLM on preference data without a reward model.\n")

    print("How it works:")
    print("  • Take preference pairs: (prompt, chosen, rejected)")
    print("  • Directly train LLM to prefer 'chosen' over 'rejected'")
    print("  • No separate reward model needed")
    print("  • No reinforcement learning needed\n")

    print("Training data format:")
    print("  Prompt:   'Explain quantum computing'")
    print("  Chosen:   '[clear, accurate explanation]'")
    print("  Rejected: '[confusing or incorrect explanation]'\n")

    print("DPO vs RLHF:\n")

    print("DPO:")
    print("  ✓ Much simpler to implement")
    print("  ✓ More stable training")
    print("  ✓ No reward model needed")
    print("  ✓ Faster training")
    print("  ✗ May be slightly less effective than RLHF")
    print("  ✗ Requires good preference pairs\n")

    print("RLHF:")
    print("  ✓ Potentially better alignment")
    print("  ✓ More flexible (can change reward function)")
    print("  ✗ Complex to implement")
    print("  ✗ Requires reward model")
    print("  ✗ Unstable training (RL is hard)")


def comparison_table():
    """Compare SFT, RLHF, and DPO."""
    print("\n" + "=" * 70)
    print("=== COMPARISON: SFT vs RLHF vs DPO ===\n")

    print("Aspect           SFT              RLHF              DPO")
    print("-" * 70)
    print("Complexity       Simple           Complex           Moderate")
    print("Data needed      10K-100K pairs   10K+ preferences  10K+ preferences")
    print("Training time    Hours-Days       Days-Weeks        Hours-Days")
    print("Stability        High             Low               High")
    print("Alignment        Moderate         Best              Good")
    print("Cost             Low              High              Moderate")
    print("Reward model?    No               Yes               No")
    print("RL required?     No               Yes (PPO)         No")
    print("-" * 70)

    print("\n\nWhen to use each:\n")

    print("SFT:")
    print("  • First step in instruction-tuning")
    print("  • Clear input-output examples")
    print("  • Limited resources")
    print("  • Prototyping\n")

    print("RLHF:")
    print("  • Maximum alignment quality")
    print("  • Have large budget and team")
    print("  • Production systems (ChatGPT, Claude)")
    print("  • Research settings\n")

    print("DPO:")
    print("  • Good alignment without complexity")
    print("  • Have preference data")
    print("  • Limited compute")
    print("  • Modern recommended approach")


def preference_data_collection():
    """Explain how to collect preference data."""
    print("\n" + "=" * 70)
    print("=== COLLECTING PREFERENCE DATA ===\n")

    print("Method 1: Human Ranking")
    print("  1. Generate 2-4 responses per prompt")
    print("  2. Humans rank from best to worst")
    print("  3. Create pairs: (best vs worst), (best vs 2nd), etc.\n")

    print("Method 2: AI Feedback (Constitutional AI)")
    print("  1. Generate multiple responses")
    print("  2. Use strong model (GPT-4) to rank")
    print("  3. Cheaper than human annotation")
    print("  4. Can introduce model biases\n")

    print("Method 3: Implicit Feedback")
    print("  1. Track user behavior (thumbs up/down, regenerations)")
    print("  2. Infer preferences from interactions")
    print("  3. Scalable but noisy\n")

    print("Best practices:")
    print("  ✓ Diverse prompts covering different tasks")
    print("  ✓ Clear ranking guidelines for annotators")
    print("  ✓ Multiple annotators per comparison")
    print("  ✓ Quality control and inter-annotator agreement")
    print("  ✓ Balance across difficulty levels")


def dpo_code_example():
    """Show conceptual DPO training code."""
    print("\n" + "=" * 70)
    print("=== DPO CODE EXAMPLE (CONCEPTUAL) ===\n")

    print("Using the TRL (Transformer Reinforcement Learning) library:\n")

    code = '''from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Prepare preference dataset
# Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_preference_dataset("preferences.jsonl")

# 3. Configure DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create reference model automatically
    beta=0.1,        # KL penalty coefficient
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256,
)

# 4. Train
dpo_trainer.train()

# 5. Save aligned model
model.save_pretrained("./aligned_model")
'''

    print(code)


def alignment_best_practices():
    """Best practices for alignment training."""
    print("\n" + "=" * 70)
    print("=== ALIGNMENT BEST PRACTICES ===\n")

    practices = [
        ("Start with SFT", "Always do supervised fine-tuning first"),
        ("High-quality preferences", "Quality matters more than quantity"),
        ("Diverse prompts", "Cover all use cases and edge cases"),
        ("KL regularization", "Prevent model from diverging too much"),
        ("Monitor during training", "Watch for reward hacking or collapse"),
        ("Validate on held-out set", "Check generalization to new prompts"),
        ("Safety considerations", "Test for harmful outputs after alignment"),
        ("Iterate", "Alignment is often done in multiple rounds"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def practical_recommendations():
    """Practical recommendations for practitioners."""
    print("=" * 70)
    print("=== PRACTICAL RECOMMENDATIONS ===\n")

    print("For most practitioners:\n")

    print("1. Start with a pre-aligned model")
    print("   Use Llama-2-chat, Mistral-instruct, etc.")
    print("   These already have SFT + alignment\n")

    print("2. If you need customization:")
    print("   a. Try prompt engineering first")
    print("   b. Then try SFT on your task")
    print("   c. Only do alignment if SFT isn't enough\n")

    print("3. If you do alignment:")
    print("   • Use DPO, not RLHF (simpler, stable)")
    print("   • Start with 1K-5K preference pairs")
    print("   • Use strong model (GPT-4) to generate preferences")
    print("   • Monitor quality closely\n")

    print("4. Resources:")
    print("   • TRL library: https://github.com/huggingface/trl")
    print("   • OpenAI's alignment research: https://openai.com/research")
    print("   • Anthropic's Constitutional AI paper")


def future_directions():
    """Discuss future directions in alignment."""
    print("\n" + "=" * 70)
    print("=== FUTURE DIRECTIONS ===\n")

    print("Emerging techniques:")
    print("  • RLAIF (RL from AI Feedback) - Use AI instead of humans")
    print("  • Constitutional AI - Explicit principles for alignment")
    print("  • Debate methods - Models argue, judges pick winner")
    print("  • Recursive reward modeling - Self-improvement loops")
    print("  • Weak-to-strong generalization - Weak models supervise strong ones\n")

    print("Open research questions:")
    print("  • How to align very powerful AI systems?")
    print("  • How to handle conflicting human preferences?")
    print("  • How to ensure robustness of alignment?")
    print("  • How to make alignment computationally cheaper?")


if __name__ == "__main__":
    llm_training_pipeline()
    supervised_fine_tuning()
    rlhf_explained()
    dpo_explained()
    comparison_table()
    preference_data_collection()
    dpo_code_example()
    alignment_best_practices()
    practical_recommendations()
    future_directions()

    print("\n" + "=" * 70)
    print("\nKey insight: Alignment is crucial for safe, helpful AI")
    print("DPO makes alignment accessible - no need for complex RLHF!")

"""
03 - LoRA Basics
=================
Parameter-efficient fine-tuning with LoRA and QLoRA.

Key concept: LoRA (Low-Rank Adaptation) fine-tunes only a small number of parameters, making fine-tuning faster and more memory-efficient.

Book reference: AI_eng.7, hands_on_LLM.III.12
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def lora_fundamentals():
    """Explain LoRA fundamentals."""
    print("=== LoRA FUNDAMENTALS ===\n")

    print("What is LoRA?")
    print("  Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method")
    print("  that adds small trainable 'adapter' layers to a frozen pre-trained model.\n")

    print("Key idea:")
    print("  Instead of updating all model weights (billions of parameters),")
    print("  LoRA trains small matrices that adapt the frozen weights.\n")

    print("Math (simplified):")
    print("  Original weight: W (n × m)")
    print("  LoRA decomposition: W + ΔW = W + B×A")
    print("  Where B (n × r) and A (r × m), with r << n, m")
    print("  r = rank (typically 8, 16, or 32)\n")

    print("Benefits:")
    print("  ✓ Trains only ~0.1-1% of parameters")
    print("  ✓ Much faster training")
    print("  ✓ Lower memory requirements")
    print("  ✓ Easy to switch between adapters")
    print("  ✓ Can merge adapter back into model")


def lora_vs_full_finetuning():
    """Compare LoRA to full fine-tuning."""
    print("\n" + "=" * 70)
    print("=== LoRA vs FULL FINE-TUNING ===\n")

    print("7B Parameter Model Example:\n")

    print("Full Fine-tuning:")
    print("  Trainable parameters: 7,000,000,000 (100%)")
    print("  GPU memory: ~80GB (for training)")
    print("  Training time: Days to weeks")
    print("  Storage per checkpoint: ~28GB")
    print("  Risk: Catastrophic forgetting\n")

    print("LoRA (r=16):")
    print("  Trainable parameters: ~20,000,000 (~0.3%)")
    print("  GPU memory: ~20GB (for training)")
    print("  Training time: Hours to days")
    print("  Storage per checkpoint: ~100MB")
    print("  Risk: Minimal forgetting\n")

    print("When to use each:")
    print("  Full fine-tuning:")
    print("    - Need maximum performance")
    print("    - Have large dataset (10K+ examples)")
    print("    - Have compute budget")
    print("    - Starting from scratch\n")

    print("  LoRA:")
    print("    - Limited compute/memory")
    print("    - Smaller dataset (500-10K examples)")
    print("    - Need fast iteration")
    print("    - Want to preserve base model knowledge")


def lora_hyperparameters():
    """Explain LoRA hyperparameters."""
    print("\n" + "=" * 70)
    print("=== LoRA HYPERPARAMETERS ===\n")

    params = [
        ("r (rank)", "8-32", "Controls adapter size. Higher = more capacity but slower"),
        ("alpha", "16-32", "Scaling factor. Typically 2×rank. Higher = stronger adaptation"),
        ("dropout", "0.05-0.1", "Dropout for LoRA layers. Prevents overfitting"),
        ("target_modules", "q_proj, v_proj", "Which layers to apply LoRA to"),
        ("bias", "none", "Whether to train bias terms ('none', 'all', 'lora_only')")]

    print("Key hyperparameters:\n")
    for param, default, description in params:
        print(f"{param:20} {default:15} {description}")

    print("\n\nCommon configurations:\n")

    configs = [
        ("Small task", "r=8, alpha=16", "Simple tasks, small datasets"),
        ("Standard", "r=16, alpha=32", "Most use cases"),
        ("Complex task", "r=32, alpha=64", "Complex domain adaptation"),
        ("QLoRA", "r=64, alpha=16", "4-bit quantized training")]

    for name, config, use_case in configs:
        print(f"{name:15} {config:20} → {use_case}")


def qlora_explained():
    """Explain QLoRA (Quantized LoRA)."""
    print("\n" + "=" * 70)
    print("=== QLoRA (QUANTIZED LoRA) ===\n")

    print("What is QLoRA?")
    print("  QLoRA combines LoRA with 4-bit quantization to reduce memory further.\n")

    print("How it works:")
    print("  1. Load base model in 4-bit precision (quantized)")
    print("  2. Keep LoRA adapters in full precision (16-bit)")
    print("  3. Train only the LoRA adapters\n")

    print("Memory savings:")
    print("  7B model without QLoRA: ~14GB (fp16)")
    print("  7B model with QLoRA: ~4GB (4-bit)")
    print("  → Can train 7B models on consumer GPUs!\n")

    print("Trade-offs:")
    print("  ✓ 4× memory reduction")
    print("  ✓ Enables larger models on small GPUs")
    print("  ✓ Minimal accuracy loss")
    print("  ✗ Slightly slower training")
    print("  ✗ More complex setup")


def lora_code_example():
    """Show conceptual LoRA code example."""
    print("\n" + "=" * 70)
    print("=== LoRA CODE EXAMPLE (CONCEPTUAL) ===\n")

    print("Using the 'peft' library:\n")

    code = '''from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,                           # Rank
    lora_alpha=32,                  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply to attention layers
    lora_dropout=0.05,              # Dropout
    bias="none",                    # Don't train bias
    task_type="CAUSAL_LM"           # Task type
)

# 3. Create LoRA model
model = get_peft_model(model, lora_config)

# 4. Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06%

# 5. Train (using standard training loop)
# trainer.train()

# 6. Save adapter only (not full model)
model.save_pretrained("./lora_adapter")
'''

    print(code)


def qlora_code_example():
    """Show conceptual QLoRA code example."""
    print("\n" + "=" * 70)
    print("=== QLoRA CODE EXAMPLE (CONCEPTUAL) ===\n")

    print("Using QLoRA with 4-bit quantization:\n")

    code = '''from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",           # Quantization type
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation dtype
    bnb_4bit_use_double_quant=True       # Double quantization
)

# 2. Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"                    # Auto device placement
)

# 3. Configure LoRA (often use higher rank with QLoRA)
lora_config = LoraConfig(
    r=64,                                # Higher rank for quantized model
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Add LoRA adapters
model = get_peft_model(model, lora_config)

# Now ready to train on consumer GPU!
'''

    print(code)


def lora_best_practices():
    """Show LoRA best practices."""
    print("\n" + "=" * 70)
    print("=== LoRA BEST PRACTICES ===\n")

    practices = [
        ("Start small", "Begin with r=8 or r=16, increase if needed"),
        ("Target key modules", "Focus on attention layers (q_proj, v_proj)"),
        ("Monitor overfitting", "Watch validation loss, use dropout if needed"),
        ("Experiment with alpha", "Try alpha = 2×rank as starting point"),
        ("Save adapters only", "Store just the adapter weights (~100MB)"),
        ("Merge for inference", "Merge adapter into base model for deployment"),
        ("Use QLoRA for large models", "Enable training 13B+ models on single GPU"),
        ("Keep learning rate low", "1e-4 to 3e-4 typically works well")]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def lora_limitations():
    """Discuss LoRA limitations."""
    print("=" * 70)
    print("=== LoRA LIMITATIONS ===\n")

    limitations = [
        "Not as powerful as full fine-tuning for very different domains",
        "May struggle with teaching completely new knowledge",
        "Requires choosing good hyperparameters (r, alpha)",
        "Can't modify all aspects of model behavior",
        "May need higher rank for complex tasks",
        "Inference slightly slower than merged model"]

    print("Limitations:")
    for limitation in limitations:
        print(f"  • {limitation}")

    print("\n\nWhen full fine-tuning might be better:")
    print("  • Completely new domain (e.g., rare language)")
    print("  • Very large dataset (100K+ examples)")
    print("  • Maximum performance critical")
    print("  • Teaching fundamentally new skills")


def practical_workflow():
    """Show practical LoRA workflow."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL LoRA WORKFLOW ===\n")

    steps = [
        "1. Prepare training data (JSONL format)",
        "2. Choose base model (Llama, Mistral, etc.)",
        "3. Configure LoRA (start with r=16, alpha=32)",
        "4. Set up training (learning rate, batch size, epochs)",
        "5. Train and monitor validation loss",
        "6. Test adapter on validation set",
        "7. Adjust hyperparameters if needed",
        "8. Save adapter weights",
        "9. (Optional) Merge adapter into base model",
        "10. Deploy for inference"]

    for step in steps:
        print(f"  {step}")

    print("\n\nTips for success:")
    print("  • Start with small experiments (100-500 examples)")
    print("  • Monitor training closely (loss, samples)")
    print("  • Compare outputs to base model")
    print("  • Keep adapters organized by version")


if __name__ == "__main__":
    lora_fundamentals()
    lora_vs_full_finetuning()
    lora_hyperparameters()
    qlora_explained()
    lora_code_example()
    qlora_code_example()
    lora_best_practices()
    lora_limitations()
    practical_workflow()

    print("\n" + "=" * 70)
    print("\nKey insight: LoRA democratizes fine-tuning!")
    print("You can now fine-tune 7B models on a single consumer GPU")

"""
04 - Model Quantization
========================
Run models locally with reduced precision for faster inference.

Key concept: Quantization reduces model size and memory by using lower precision (8-bit, 4-bit) instead of 16-bit, with minimal accuracy loss.

Book reference: AI_eng.7, hands_on_LLM.III.12
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def quantization_fundamentals():
    """Explain quantization basics."""
    print("=== QUANTIZATION FUNDAMENTALS ===\n")

    print("What is quantization?")
    print("  Reducing the precision of model weights from high precision (float32/float16)")
    print("  to lower precision (int8/int4) to save memory and speed up inference.\n")

    print("Precision formats:")
    print("  • float32 (FP32): 32 bits per parameter - Original training format")
    print("  • float16 (FP16): 16 bits per parameter - Standard inference format")
    print("  • bfloat16 (BF16): 16 bits per parameter - Better for training")
    print("  • int8: 8 bits per parameter - 2× memory savings vs FP16")
    print("  • int4: 4 bits per parameter - 4× memory savings vs FP16\n")

    print("Memory comparison (7B parameter model):")
    print("  FP32:  28 GB")
    print("  FP16:  14 GB")
    print("  INT8:   7 GB  (2× reduction)")
    print("  INT4:   3.5 GB (4× reduction)")


def quantization_types():
    """Explain different quantization approaches."""
    print("\n" + "=" * 70)
    print("=== QUANTIZATION TYPES ===\n")

    print("1. Post-Training Quantization (PTQ)")
    print("   Convert trained model to lower precision")
    print("   ✓ No retraining needed")
    print("   ✓ Fast to apply")
    print("   ✗ Some accuracy loss\n")

    print("2. Quantization-Aware Training (QAT)")
    print("   Train model with quantization in mind")
    print("   ✓ Better accuracy retention")
    print("   ✗ Requires retraining")
    print("   ✗ More complex\n")

    print("3. Dynamic Quantization")
    print("   Quantize weights, keep activations in FP16")
    print("   ✓ Good accuracy")
    print("   ✓ Easy to apply")
    print("   ✗ Less memory savings\n")

    print("4. Static Quantization")
    print("   Quantize both weights and activations")
    print("   ✓ Maximum memory savings")
    print("   ✓ Fastest inference")
    print("   ✗ Requires calibration data")


def int8_quantization():
    """Explain 8-bit quantization."""
    print("\n" + "=" * 70)
    print("=== 8-BIT QUANTIZATION ===\n")

    print("How it works:")
    print("  Map float16 values [-∞, +∞] to int8 [-127, 127]\n")

    print("Example mapping:")
    print("  FP16 range: [-2.5, 2.5]")
    print("  Scale factor: 2.5 / 127 ≈ 0.0197")
    print("  FP16: -2.5  → INT8: -127")
    print("  FP16:  0.0  → INT8:    0")
    print("  FP16: +2.5  → INT8: +127\n")

    print("Benefits:")
    print("  ✓ 2× memory reduction vs FP16")
    print("  ✓ 2-3× faster inference on some hardware")
    print("  ✓ Minimal accuracy loss (<1% typically)")
    print("  ✓ Good balance of speed and quality\n")

    print("Use cases:")
    print("  • Production inference on limited GPU memory")
    print("  • Serving multiple models on same GPU")
    print("  • Mobile/edge deployment")


def int4_quantization():
    """Explain 4-bit quantization."""
    print("\n" + "=" * 70)
    print("=== 4-BIT QUANTIZATION ===\n")

    print("How it works:")
    print("  Map float16 values to just 16 discrete levels (-8 to 7)\n")

    print("Advanced techniques:")
    print("  • NF4 (NormalFloat4): Optimized for normal distribution")
    print("  • Group quantization: Different scales for parameter groups")
    print("  • Double quantization: Quantize the quantization scales\n")

    print("Benefits:")
    print("  ✓ 4× memory reduction vs FP16")
    print("  ✓ Enable large models on consumer hardware")
    print("  ✓ 3.5GB for 7B model (fits on many GPUs)")
    print("  ✗ More accuracy loss than INT8 (1-3%)")
    print("  ✗ Slower than INT8 on some hardware\n")

    print("Use cases:")
    print("  • Running large models locally (13B, 30B)")
    print("  • Fine-tuning with QLoRA")
    print("  • Consumer GPU inference")


def quantization_code_example():
    """Show conceptual quantization code."""
    print("\n" + "=" * 70)
    print("=== QUANTIZATION CODE EXAMPLE ===\n")

    print("8-bit quantization with bitsandbytes:\n")

    code_8bit = '''from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,         # Threshold for outlier detection
    llm_int8_skip_modules=None,     # Modules to keep in FP16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"                # Auto distribute across GPUs
)

# Model now uses ~7GB instead of ~14GB
'''

    print(code_8bit)

    print("\n4-bit quantization with NF4:\n")

    code_4bit = '''# Configure 4-bit quantization with NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True       # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Model now uses ~3.5GB instead of ~14GB!
'''

    print(code_4bit)


def quantization_comparison():
    """Compare different quantization levels."""
    print("\n" + "=" * 70)
    print("=== QUANTIZATION COMPARISON ===\n")

    print("7B Parameter Model:\n")

    print("Format       Memory    Perplexity   Speed     Use Case")
    print("-" * 70)
    print("FP16         14 GB     10.0 (base)  1.0×      High-end GPUs")
    print("INT8          7 GB     10.1 (+1%)   1.5-2×    Production inference")
    print("INT4 (NF4)    3.5 GB   10.3 (+3%)   1.2-1.5×  Consumer GPUs")
    print("-" * 70)

    print("\n\n13B Parameter Model:\n")

    print("Format       Memory    Feasibility")
    print("-" * 70)
    print("FP16         26 GB     High-end GPU only (A100)")
    print("INT8         13 GB     Mid-range GPU (3090, 4090)")
    print("INT4 (NF4)    6.5 GB   Consumer GPU (RTX 3060 12GB)")
    print("-" * 70)


def hardware_considerations():
    """Discuss hardware for quantized models."""
    print("\n" + "=" * 70)
    print("=== HARDWARE CONSIDERATIONS ===\n")

    print("GPU VRAM requirements for popular models:\n")

    models = [
        ("7B params", "14 GB (FP16)", "7 GB (INT8)", "3.5 GB (INT4)"),
        ("13B params", "26 GB (FP16)", "13 GB (INT8)", "6.5 GB (INT4)"),
        ("30B params", "60 GB (FP16)", "30 GB (INT8)", "15 GB (INT4)"),
        ("65B params", "130 GB (FP16)", "65 GB (INT8)", "32.5 GB (INT4)"),
    ]

    print("Model       FP16          INT8         INT4")
    print("-" * 65)
    for model, fp16, int8, int4 in models:
        print(f"{model:12} {fp16:13} {int8:12} {int4:12}")

    print("\n\nRecommended GPUs:\n")

    gpus = [
        ("RTX 3090", "24 GB", "7B INT4, 13B INT4"),
        ("RTX 4090", "24 GB", "7B INT8, 13B INT4"),
        ("A100 (40GB)", "40 GB", "7B FP16, 13B INT8, 30B INT4"),
        ("A100 (80GB)", "80 GB", "13B FP16, 30B INT8, 65B INT4"),
    ]

    print("GPU          VRAM      Can Run")
    print("-" * 65)
    for gpu, vram, can_run in gpus:
        print(f"{gpu:15} {vram:10} {can_run}")


def quantization_accuracy_tips():
    """Tips for maintaining accuracy with quantization."""
    print("\n" + "=" * 70)
    print("=== MAINTAINING ACCURACY ===\n")

    tips = [
        ("Use NF4 for 4-bit", "NF4 optimized for neural network weight distributions"),
        ("Enable double quantization", "Quantize the quantization scales for better quality"),
        ("Keep compute in BF16", "Do computations in BF16 even with 4-bit weights"),
        ("Test on validation set", "Measure actual accuracy impact on your task"),
        ("Calibrate with representative data", "Use relevant data for quantization calibration"),
        ("Consider mixed precision", "Keep important layers in higher precision"),
        ("Start with 8-bit", "Try INT8 first, only use INT4 if memory constrained"),
    ]

    for tip, explanation in tips:
        print(f"✓ {tip}")
        print(f"  → {explanation}\n")


def quantization_pitfalls():
    """Common quantization pitfalls."""
    print("=" * 70)
    print("=== COMMON PITFALLS ===\n")

    pitfalls = [
        ("Quantizing without testing",
         "Always validate accuracy on your specific task"),

        ("Using aggressive quantization unnecessarily",
         "INT8 often sufficient, INT4 only when memory limited"),

        ("Ignoring outlier features",
         "Some values hard to quantize - use mixed precision"),

        ("Quantizing wrong model parts",
         "Embeddings and output layer often need higher precision"),

        ("Not considering inference speed",
         "Heavily quantized models may be slower on some hardware"),

        ("Forgetting calibration",
         "Static quantization needs calibration data"),
    ]

    for pitfall, solution in pitfalls:
        print(f"✗ {pitfall}")
        print(f"  → {solution}\n")


def quantization_best_practices():
    """Best practices for quantization."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Start with FP16 baseline - Measure accuracy before quantization",
        "2. Try INT8 first - Good balance of speed and quality",
        "3. Use INT4 for memory constraints - When you need to fit in limited VRAM",
        "4. Always validate - Test on your specific task and data",
        "5. Monitor perplexity - Track model perplexity as quality metric",
        "6. Use NF4 for 4-bit - Better than regular 4-bit quantization",
        "7. Enable double quantization - Minor memory cost, better quality",
        "8. Benchmark inference - Measure actual speed gains on your hardware",
        "9. Keep calibration data representative - Use real task data",
        "10. Document quantization settings - Track config for reproducibility",
    ]

    for practice in practices:
        print(f"  {practice}")


def practical_workflow():
    """Show practical quantization workflow."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL WORKFLOW ===\n")

    print("Decision tree:\n")

    print("""
    START
      │
      ├─ Do you have 24GB+ VRAM?
      │   └─ YES → Use INT8 for best quality/speed balance
      │
      ├─ Do you have <24GB VRAM?
      │   └─ YES → Use INT4 (NF4) to fit model
      │
      ├─ Is accuracy critical?
      │   └─ YES → Start with INT8, measure accuracy loss
      │
      └─ Need to run large model (30B+)?
          └─ YES → INT4 is your only option on consumer GPU
    """)

    print("\nSteps:")
    print("  1. Load model in FP16, test baseline accuracy")
    print("  2. Apply quantization (INT8 or INT4)")
    print("  3. Test accuracy on validation set")
    print("  4. If accuracy acceptable, benchmark inference speed")
    print("  5. Deploy quantized model")


if __name__ == "__main__":
    quantization_fundamentals()
    quantization_types()
    int8_quantization()
    int4_quantization()
    quantization_code_example()
    quantization_comparison()
    hardware_considerations()
    quantization_accuracy_tips()
    quantization_pitfalls()
    quantization_best_practices()
    practical_workflow()

    print("\n" + "=" * 70)
    print("\nKey insight: Quantization enables running large models on consumer hardware")
    print("INT4 makes 7B models fit in 3.5GB - accessible to everyone!")

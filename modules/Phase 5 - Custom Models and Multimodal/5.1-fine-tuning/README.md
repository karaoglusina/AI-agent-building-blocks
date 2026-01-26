# Module 5.1: Fine-tuning LLMs

> *"Customize models for your specific task and domain"*

This module covers fine-tuning Large Language Models - when to fine-tune, how to prepare data, parameter-efficient methods (LoRA/QLoRA), quantization, alignment techniques, and proper evaluation.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_when_to_finetune.py` | When to Fine-tune | Decision framework: fine-tune vs RAG vs prompt engineering |
| `02_data_preparation.py` | Training Data Prep | Format data, data quality, data synthesis basics |
| `03_lora_basics.py` | LoRA Basics | Parameter-efficient fine-tuning with LoRA/QLoRA |
| `04_quantization.py` | Model Quantization | Run models locally with reduced precision |
| `05_sft_rlhf_dpo_overview.py` | SFT/RLHF/DPO Overview | Understand alignment techniques (concepts) |
| `06_evaluation.py` | Fine-tune Evaluation | Measure improvement |

## Why Fine-tuning?

Fine-tuning adapts a pre-trained model to your specific task or domain:
- **Specialization**: Teach domain-specific knowledge (legal, medical)
- **Consistency**: More reliable output format and style
- **Efficiency**: Smaller fine-tuned models can outperform larger base models
- **Cost Reduction**: Reduce prompt tokens, lower latency
- **Customization**: Teach specific behaviors and preferences

## Core Concepts

### 1. The Fine-tuning Hierarchy

```
Most Problems                      Fewest Problems
     ↓                                   ↓
Prompt Engineering → RAG → Fine-tuning → Pre-training
     ↑                                   ↑
Cheapest/Fastest                    Most Expensive
```

**Decision flow:**
1. Try prompt engineering first (zero cost)
2. Add RAG if you need external knowledge
3. Fine-tune if you need consistency/specialization
4. Pre-train only for completely new domains

### 2. LoRA: Parameter-Efficient Fine-tuning

Traditional fine-tuning updates all parameters. LoRA trains small adapter matrices:

```
Original:  W (millions of parameters)
LoRA:      W + B×A (thousands of parameters)

Result: 0.1-1% of parameters trained, 90% memory saved
```

**Benefits:**
- Train 7B models on consumer GPUs
- Fast iteration (hours instead of days)
- Easy to switch between adapters
- Minimal risk of catastrophic forgetting

### 3. Quantization for Efficiency

```
Format      Memory (7B)    Quality      Use Case
────────────────────────────────────────────────
FP16        14 GB          Best         High-end GPUs
INT8         7 GB          -1%          Production
INT4         3.5 GB        -3%          Consumer GPUs
```

**QLoRA**: Combine LoRA + 4-bit quantization
- Train 7B models in ~5GB VRAM
- Enables local training on RTX 3090, 4090
- Democratizes fine-tuning

### 4. Training Data Quality

```python
# Good training example
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

**Quality over quantity:**
- 500 high-quality examples > 5000 noisy examples
- Consistent formatting is critical
- Diverse examples prevent overfitting
- Validate and clean your data

### 5. Alignment Techniques

**SFT (Supervised Fine-Tuning):**
- Train on (instruction, response) pairs
- Teaches instruction-following
- Straightforward supervised learning

**RLHF (Reinforcement Learning from Human Feedback):**
- Train reward model from human preferences
- Use RL (PPO) to optimize LLM
- Used by ChatGPT, Claude
- Complex but powerful

**DPO (Direct Preference Optimization):**
- Directly optimize on preference pairs
- Simpler than RLHF, no reward model
- Modern recommended approach
- Good balance of simplicity and effectiveness

```
Complexity:    SFT < DPO < RLHF
Effectiveness: SFT < DPO ≈ RLHF
```

## Running the Examples

Each script demonstrates different aspects of fine-tuning:

```bash
# When to fine-tune (decision framework)
python modules/phase5/5.1-fine-tuning/01_when_to_finetune.py

# Data preparation best practices
python modules/phase5/5.1-fine-tuning/02_data_preparation.py

# LoRA and QLoRA concepts
python modules/phase5/5.1-fine-tuning/03_lora_basics.py

# Quantization for efficient inference
python modules/phase5/5.1-fine-tuning/04_quantization.py

# Alignment techniques (SFT/RLHF/DPO)
python modules/phase5/5.1-fine-tuning/05_sft_rlhf_dpo_overview.py

# Evaluation methods
python modules/phase5/5.1-fine-tuning/06_evaluation.py
```

## Practical Fine-tuning Workflow

### 1. Decide if Fine-tuning is Needed

```python
# Use this decision tree
if examples < 50:
    use_prompt_engineering()
elif need_latest_info:
    use_rag()
elif have_500_plus_examples and need_consistency:
    use_finetuning()
else:
    use_prompt_engineering_or_rag()
```

### 2. Prepare Training Data

```python
# Collect data
examples = collect_instruction_response_pairs()  # 500-5000 examples

# Clean and format
examples = validate_and_clean(examples)
train, val = split_data(examples, test_size=0.15)

# Save in JSONL format
with open("train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")
```

### 3. Configure LoRA

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank (start with 16)
    lora_alpha=32,             # 2× rank
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

### 4. Train

```python
# Use Transformers Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 5. Evaluate

```python
# Automatic metrics
perplexity = evaluate_perplexity(model, test_set)
accuracy = evaluate_accuracy(model, test_set)

# Human evaluation
ratings = get_human_ratings(model, test_prompts)

# Compare to baseline
print(f"Base: {base_perplexity} → Fine-tuned: {perplexity}")
```

### 6. Deploy

```python
# Save adapter (not full model)
model.save_pretrained("./my_adapter")  # ~100MB

# Or merge for deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

## Best Practices

### 1. When to Fine-tune
- **Do fine-tune** when:
  - Need consistent output format
  - Have 500+ quality examples
  - Domain-specific language (legal, medical)
  - Base model fails at task
  - Need lower latency

- **Don't fine-tune** when:
  - <50 examples (use prompt engineering)
  - Need latest info (use RAG)
  - Task is general-purpose
  - Base model already works well

### 2. Data Preparation
- **Quality > Quantity**: 500 perfect examples beat 5000 noisy ones
- **Diverse examples**: Cover all use cases and edge cases
- **Consistent format**: Standardize all training data
- **Clean thoroughly**: Remove errors, duplicates, outliers
- **Validate splits**: Ensure train/val/test have no leakage

### 3. LoRA Configuration
- **Start small**: r=8 or r=16, increase if needed
- **Target attention**: Focus on q_proj, v_proj
- **Alpha = 2×rank**: Good starting point (r=16, alpha=32)
- **Monitor overfitting**: Use dropout if validation loss increases
- **Save adapters**: Store just adapter weights (~100MB)

### 4. Quantization
- **INT8 first**: Try 8-bit before 4-bit
- **Use NF4**: For 4-bit, NF4 is best
- **Enable double quant**: Small cost, better quality
- **Validate accuracy**: Always test on your data
- **QLoRA for training**: Enables 7B training on consumer GPUs

### 5. Evaluation
- **Multiple metrics**: Perplexity, accuracy, F1, human ratings
- **Compare to baseline**: Always measure improvement
- **Error analysis**: Review failures and successes
- **Hold out test set**: Never train on test data
- **Monitor production**: Track metrics over time

### 6. Training
- **Low learning rate**: 1e-4 to 3e-4 for LoRA
- **Small batch size**: 4-8 with gradient accumulation
- **Few epochs**: 3-5 epochs typically enough
- **Monitor validation**: Stop if overfitting
- **Save checkpoints**: Keep multiple versions

## Common Pitfalls

### 1. Training on Test Data
**Problem**: Inflated metrics, poor generalization
**Solution**: Strict train/val/test separation, check for duplicates

### 2. Too Few Examples
**Problem**: Overfitting, poor performance
**Solution**: Collect more data or use prompt engineering

### 3. Low-Quality Data
**Problem**: Model learns bad patterns
**Solution**: Manual review, data cleaning, validation

### 4. Wrong Hyperparameters
**Problem**: Poor convergence or overfitting
**Solution**: Start with proven configs, tune carefully

### 5. No Evaluation Plan
**Problem**: Can't measure if fine-tuning helped
**Solution**: Define metrics before training, compare to baseline

### 6. Ignoring Base Model Quality
**Problem**: Fine-tuning can't fix fundamental model issues
**Solution**: Start with strong base model (Llama-2, Mistral, etc.)

## Hardware Requirements

### Training (with QLoRA)
```
Model Size    VRAM Needed    Example GPU
──────────────────────────────────────────
7B            5-6 GB         RTX 3060 12GB
13B           10-12 GB       RTX 3090 24GB
30B           20-24 GB       RTX 4090 24GB
65B           40-48 GB       A100 80GB
```

### Inference (with quantization)
```
Model Size    INT8 VRAM     INT4 VRAM
────────────────────────────────────────
7B            7 GB          3.5 GB
13B           13 GB         6.5 GB
30B           30 GB         15 GB
65B           65 GB         32.5 GB
```

## Libraries and Tools

### Core Libraries
```bash
pip install transformers  # Hugging Face models
pip install peft          # LoRA and other PEFT methods
pip install bitsandbytes  # Quantization
pip install datasets      # Dataset loading
pip install trl           # RLHF/DPO training
```

### Training Frameworks
- **Hugging Face Transformers**: Standard library for LLM training
- **TRL** (Transformer Reinforcement Learning): DPO and RLHF
- **Axolotl**: Simplified fine-tuning framework
- **LLaMA-Factory**: Easy fine-tuning of LLaMA models

### Model Hubs
- **Hugging Face Hub**: Pre-trained models and datasets
- **OpenLLM Leaderboard**: Compare model performance
- **Together AI**: Fine-tuning as a service

## Real-World Examples

### Example 1: Customer Support Classification
```
Task: Classify support tickets into 10 categories
Data: 2000 labeled tickets
Approach: LoRA fine-tuning of Llama-2-7b
Results: 68% → 87% accuracy (+19%)
Training: 2 hours on RTX 3090
```

### Example 2: Code Generation (Company Style)
```
Task: Generate code following company style guide
Data: 1500 code examples with comments
Approach: QLoRA fine-tuning of CodeLlama-13b
Results: More consistent formatting, fewer errors
Training: 4 hours on RTX 4090
```

### Example 3: Medical Report Summarization
```
Task: Summarize clinical reports
Data: 5000 report/summary pairs
Approach: Full fine-tuning of Llama-2-13b
Results: ROUGE-L 0.32 → 0.51 (+59%)
Training: 2 days on A100
```

## Cost Considerations

### OpenAI Fine-tuning (GPT-3.5)
- Training: $0.008 per 1K tokens
- Inference: $0.012 per 1K tokens (1.5× base cost)
- 1M tokens training ≈ $8

### Self-hosted Fine-tuning
- GPU rental: $1-3/hour (depending on GPU)
- Typical training: 2-8 hours
- One-time cost: $2-25 per model
- Inference: Free (after GPU cost)

### When to Use Each
- **OpenAI**: Quick experiments, no GPU access
- **Self-hosted**: Many experiments, repeated use, custom models

## Book References

- `AI_eng.7` - Fine-tuning and customization
- `AI_eng.8` - Data preparation and quality
- `hands_on_LLM.III.12` - LoRA, QLoRA, and PEFT
- `speach_lang.I.12.7` - Alignment and RLHF

## Next Steps

After mastering fine-tuning:
- Module 5.2: Custom Embeddings - Fine-tune embeddings for your domain
- Module 5.3: Advanced NLP - Extract structured information
- Module 5.4: Multimodal - Work with images and text
- Module 4.7: Cloud Deployment - Deploy fine-tuned models

## Additional Resources

- **Papers:**
  - LoRA: https://arxiv.org/abs/2106.09685
  - QLoRA: https://arxiv.org/abs/2305.14314
  - DPO: https://arxiv.org/abs/2305.18290

- **Tutorials:**
  - Hugging Face PEFT docs: https://huggingface.co/docs/peft
  - Axolotl guide: https://github.com/OpenAccess-AI-Collective/axolotl

- **Communities:**
  - r/LocalLLaMA - Fine-tuning discussions
  - Hugging Face Discord - Technical support

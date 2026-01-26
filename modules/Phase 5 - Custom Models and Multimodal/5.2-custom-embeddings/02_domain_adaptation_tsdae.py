"""
02 - Domain Adaptation with TSDAE
===================================
Unsupervised domain adaptation for embeddings using TSDAE.

Key concept: TSDAE (Transformer-based Sequential Denoising Auto-Encoder) adapts pre-trained embeddings to your domain without labeled data.

Book reference: hands_on_LLM.III.10
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def domain_adaptation_intro():
    """Introduce domain adaptation for embeddings."""
    print("=== DOMAIN ADAPTATION FOR EMBEDDINGS ===\n")

    print("The problem:")
    print("  Pre-trained embedding models work well on general text but may struggle")
    print("  with domain-specific vocabulary and concepts (medical, legal, technical).\n")

    print("Example:")
    print("  General model: 'python' → programming language OR snake")
    print("  CS domain: 'python' → programming language (high confidence)\n")

    print("Solution: Domain adaptation")
    print("  Fine-tune embedding model on your domain text without labeled data\n")

    print("Why domain adaptation?")
    print("  ✓ Improve embedding quality for your domain")
    print("  ✓ Learn domain-specific vocabulary")
    print("  ✓ Better semantic understanding")
    print("  ✓ No labeled data required (unsupervised)")
    print("  ✗ Requires domain text corpus")
    print("  ✗ Training time and compute")


def tsdae_explained():
    """Explain TSDAE method."""
    print("\n" + "=" * 70)
    print("=== TSDAE (TRANSFORMER DENOISING AUTO-ENCODER) ===\n")

    print("What is TSDAE?")
    print("  An unsupervised learning method that improves embeddings by training")
    print("  the model to reconstruct sentences from noisy versions.\n")

    print("How it works:\n")

    print("  1. Input: Original sentence")
    print("     'Machine learning is a subset of AI'\n")

    print("  2. Noise: Delete random words (60%)")
    print("     'Machine _____ is _____ subset _____ AI'\n")

    print("  3. Encode: Generate embedding of noisy sentence")
    print("     embedding = model.encode(noisy_sentence)\n")

    print("  4. Decode: Try to reconstruct original")
    print("     reconstructed = decoder(embedding)\n")

    print("  5. Loss: Measure reconstruction error")
    print("     loss = cross_entropy(reconstructed, original)\n")

    print("Training objective:")
    print("  Model learns to capture essential meaning even with missing words\n")

    print("Benefits:")
    print("  ✓ No labeled data needed")
    print("  ✓ Works with domain-specific text")
    print("  ✓ Learns domain vocabulary and concepts")
    print("  ✓ Improves semantic understanding")


def when_to_use_domain_adaptation():
    """Explain when domain adaptation is useful."""
    print("\n" + "=" * 70)
    print("=== WHEN TO USE DOMAIN ADAPTATION ===\n")

    print("Use domain adaptation when:\n")

    scenarios = [
        ("Technical domain", "Legal, medical, scientific text with jargon"),
        ("Poor base performance", "Off-the-shelf models struggle on your data"),
        ("Domain-specific vocabulary", "Unique terms not in general training data"),
        ("Have unlabeled data", "Lots of domain text but no labeled examples"),
        ("Need better search", "Semantic search not working well in your domain"),
    ]

    for scenario, description in scenarios:
        print(f"  • {scenario}: {description}")

    print("\n\nSkip domain adaptation when:\n")

    skip = [
        "General domain text (news, social media)",
        "Pre-trained model works well already",
        "Have <1000 domain examples",
        "Can't afford training time/cost",
    ]

    for reason in skip:
        print(f"  • {reason}")


def tsdae_code_example():
    """Show TSDAE training example."""
    print("\n" + "=" * 70)
    print("=== TSDAE CODE EXAMPLE ===\n")

    code = '''from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

# 1. Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare your domain text (unlabeled)
domain_texts = [
    "Neural networks consist of interconnected layers of neurons",
    "Gradient descent optimizes model parameters iteratively",
    "Backpropagation computes gradients for weight updates",
    # ... thousands more sentences from your domain
]

# 3. Create denoising dataset
# Randomly deletes 60% of words to create noisy versions
train_dataset = DenoisingAutoEncoderDataset(domain_texts)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 4. Configure TSDAE loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model,
    decoder_name_or_path='all-MiniLM-L6-v2',  # Use same model for decoding
    tie_encoder_decoder=True                   # Share encoder/decoder weights
)

# 5. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    show_progress_bar=True
)

# 6. Save adapted model
model.save('models/domain-adapted-embeddings')

# 7. Use adapted model
adapted_model = SentenceTransformer('models/domain-adapted-embeddings')
embeddings = adapted_model.encode(domain_texts)
'''

    print(code)


def evaluation_example():
    """Show how to evaluate domain adaptation."""
    print("\n" + "=" * 70)
    print("=== EVALUATING DOMAIN ADAPTATION ===\n")

    print("Method 1: Semantic Similarity")
    print("  Compare similar pairs before/after adaptation\n")

    code1 = '''from sentence_transformers import util

# Test pairs (similar concepts in your domain)
test_pairs = [
    ("neural network", "artificial neural network"),
    ("gradient descent", "optimization algorithm"),
    ("overfitting", "model memorization"),
]

# Evaluate base model
base_model = SentenceTransformer('all-MiniLM-L6-v2')
for sent1, sent2 in test_pairs:
    emb1 = base_model.encode(sent1)
    emb2 = base_model.encode(sent2)
    sim = util.cos_sim(emb1, emb2).item()
    print(f"Base: {sent1} <-> {sent2}: {sim:.3f}")

# Evaluate adapted model
adapted = SentenceTransformer('models/domain-adapted')
for sent1, sent2 in test_pairs:
    emb1 = adapted.encode(sent1)
    emb2 = adapted.encode(sent2)
    sim = util.cos_sim(emb1, emb2).item()
    print(f"Adapted: {sent1} <-> {sent2}: {sim:.3f}")

# Expected: Higher similarity for domain terms after adaptation
'''

    print(code1)

    print("\n\nMethod 2: Retrieval Performance")
    print("  Measure search quality on domain queries\n")

    code2 = '''# Your domain documents
docs = ["doc1 text...", "doc2 text...", ...]

# Domain queries with known relevant docs
queries = [
    ("query1", [0, 3]),      # Relevant doc indices
    ("query2", [1, 4, 7]),
]

def evaluate_retrieval(model, docs, queries):
    doc_embs = model.encode(docs)

    mrr_sum = 0  # Mean Reciprocal Rank
    for query, relevant_ids in queries:
        query_emb = model.encode(query)
        scores = util.cos_sim(query_emb, doc_embs)[0]

        # Check rank of first relevant doc
        ranked = scores.argsort(descending=True)
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in relevant_ids:
                mrr_sum += 1.0 / rank
                break

    return mrr_sum / len(queries)

base_mrr = evaluate_retrieval(base_model, docs, queries)
adapted_mrr = evaluate_retrieval(adapted_model, docs, queries)

print(f"Base MRR: {base_mrr:.3f}")
print(f"Adapted MRR: {adapted_mrr:.3f}")
print(f"Improvement: {(adapted_mrr - base_mrr) / base_mrr * 100:.1f}%")
'''

    print(code2)


def hyperparameters_guide():
    """Guide for TSDAE hyperparameters."""
    print("\n" + "=" * 70)
    print("=== HYPERPARAMETERS ===\n")

    params = [
        ("deletion_prob", "0.6", "Probability of deleting each word (60% recommended)"),
        ("epochs", "1-3", "Number of training epochs (1 often enough)"),
        ("batch_size", "8-32", "Batch size (depends on GPU memory)"),
        ("learning_rate", "3e-5", "Learning rate (default works well)"),
        ("warmup_steps", "100", "Number of warmup steps"),
    ]

    print("Key hyperparameters:\n")
    for param, default, description in params:
        print(f"  {param:20} {default:10} → {description}")

    print("\n\nData requirements:")
    print("  Minimum: 1,000 domain sentences")
    print("  Recommended: 10,000+ sentences")
    print("  More data → better adaptation\n")

    print("Training time:")
    print("  10K sentences: ~30 minutes (GPU)")
    print("  100K sentences: ~3 hours (GPU)")


def domain_adaptation_alternatives():
    """Show alternative domain adaptation methods."""
    print("\n" + "=" * 70)
    print("=== ALTERNATIVE METHODS ===\n")

    print("1. Continued Pre-training (MLM)")
    print("   Continue masked language modeling on domain text")
    print("   ✓ Learns domain vocabulary well")
    print("   ✗ More complex than TSDAE\n")

    print("2. Contrastive Learning (SimCSE)")
    print("   Learn by contrasting similar/dissimilar sentences")
    print("   ✓ State-of-the-art quality")
    print("   ✗ Requires careful data preparation\n")

    print("3. Supervised Fine-tuning")
    print("   Train on labeled pairs (query, relevant doc)")
    print("   ✓ Best performance")
    print("   ✗ Requires labeled data (expensive)\n")

    print("4. TSDAE (Recommended)")
    print("   Denoising auto-encoder approach")
    print("   ✓ Simple and effective")
    print("   ✓ No labeled data needed")
    print("   ✓ Works well in practice")


def real_world_example():
    """Show real-world example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE ===\n")

    print("Domain: Medical research papers\n")

    print("Problem:")
    print("  - General embedding model struggles with medical terminology")
    print("  - Poor semantic search on medical concepts")
    print("  - Example: 'myocardial infarction' not similar to 'heart attack'\n")

    print("Solution:")
    print("  1. Collect 50K sentences from medical papers (abstracts)")
    print("  2. Apply TSDAE to adapt all-mpnet-base-v2")
    print("  3. Train for 1 epoch (~2 hours on GPU)\n")

    print("Results:")
    print("  Metric                  Base Model    Adapted Model    Improvement")
    print("  " + "-" * 70)
    print("  Medical term similarity 0.52          0.71             +37%")
    print("  Retrieval MRR           0.48          0.63             +31%")
    print("  User satisfaction       65%           82%              +17%\n")

    print("Conclusion:")
    print("  Domain adaptation significantly improved medical search quality")


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Start with good base model", "all-mpnet-base-v2 adapts better than small models"),
        ("Collect diverse domain text", "Cover different aspects of your domain"),
        ("Use enough data", "10K+ sentences recommended"),
        ("Clean your data", "Remove boilerplate, duplicates, noise"),
        ("Evaluate before/after", "Quantify improvement on your task"),
        ("Save checkpoints", "Keep intermediate versions"),
        ("Test on held-out data", "Avoid overfitting to training distribution"),
        ("Consider supervised if possible", "Labeled data gives better results"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def common_mistakes():
    """Show common mistakes."""
    print("=" * 70)
    print("=== COMMON MISTAKES ===\n")

    mistakes = [
        ("Too little data",
         "Need 1K+ sentences minimum, 10K+ recommended"),

        ("Mixed domains",
         "Mixing multiple domains dilutes adaptation"),

        ("Not evaluating",
         "Must measure if adaptation actually helped"),

        ("Over-training",
         "Too many epochs can hurt generalization"),

        ("Ignoring data quality",
         "Noisy data produces worse embeddings"),

        ("Using too small base model",
         "Small models have less capacity to adapt"),
    ]

    for mistake, consequence in mistakes:
        print(f"✗ {mistake}")
        print(f"  → {consequence}\n")


def practical_workflow():
    """Show practical workflow."""
    print("=" * 70)
    print("=== PRACTICAL WORKFLOW ===\n")

    steps = [
        "1. Collect domain text (10K+ sentences from your domain)",
        "2. Clean and deduplicate text",
        "3. Evaluate base model on test queries",
        "4. Apply TSDAE for 1 epoch",
        "5. Evaluate adapted model on same test queries",
        "6. If improvement insufficient, try:",
        "   - More training data",
        "   - Better base model",
        "   - Longer training (2-3 epochs)",
        "   - Supervised fine-tuning (if labels available)",
        "7. Deploy adapted model",
        "8. Monitor quality in production",
    ]

    for step in steps:
        print(f"  {step}")


if __name__ == "__main__":
    domain_adaptation_intro()
    tsdae_explained()
    when_to_use_domain_adaptation()
    tsdae_code_example()
    evaluation_example()
    hyperparameters_guide()
    domain_adaptation_alternatives()
    real_world_example()
    best_practices()
    common_mistakes()
    practical_workflow()

    print("\n" + "=" * 70)
    print("\nKey insight: Domain adaptation improves embeddings without labeled data!")
    print("TSDAE is simple, effective, and works well for domain-specific text")

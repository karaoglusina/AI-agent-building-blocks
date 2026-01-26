"""
02 - CLIP Basics
=================
Text-image similarity with CLIP.

Key concept: CLIP (Contrastive Language-Image Pre-training) creates aligned embeddings for text and images, enabling zero-shot image classification and semantic search.

Book reference: hands_on_LLM.II.9
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def clip_intro():
    """Introduce CLIP."""
    print("=== CLIP (CONTRASTIVE LANGUAGE-IMAGE PRE-TRAINING) ===\n")

    print("What is CLIP?")
    print("  A model trained to understand relationships between images and text.")
    print("  Creates embeddings where similar text and images are close together.\n")

    print("Key capabilities:")
    print("  ✓ Zero-shot image classification")
    print("  ✓ Text-to-image search")
    print("  ✓ Image-to-text search")
    print("  ✓ Semantic similarity between images and text")
    print("  ✓ No fine-tuning required for new categories\n")

    print("How it works:")
    print("  1. Image encoder: Converts images to embeddings")
    print("  2. Text encoder: Converts text to embeddings")
    print("  3. Same embedding space: Similar concepts are close")
    print("  4. Cosine similarity: Measure relatedness\n")

    print("Use cases:")
    print("  • Image search with text queries")
    print("  • Content moderation")
    print("  • Product categorization")
    print("  • Visual question answering")


def basic_usage():
    """Show basic CLIP usage."""
    print("\n" + "=" * 70)
    print("=== BASIC USAGE ===\n")

    code = '''from PIL import Image
import clip
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image
image = Image.open("photo.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)

# Prepare text
text_inputs = clip.tokenize([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
]).to(device)

# Get embeddings
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Similarities:")
    labels = ["cat", "dog", "bird"]
    for label, score in zip(labels, similarity[0]):
        print(f"  {label}: {score.item():.2%}")

# Output:
# cat:  92.5%
# dog:   5.2%
# bird:  2.3%
'''

    print(code)


def zero_shot_classification():
    """Show zero-shot classification."""
    print("\n" + "=" * 70)
    print("=== ZERO-SHOT IMAGE CLASSIFICATION ===\n")

    code = '''def classify_image(image_path, categories):
    """
    Classify image into one of the provided categories.
    No training required!
    """
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Create text prompts
    text_prompts = [f"a photo of a {cat}" for cat in categories]
    text_inputs = clip.tokenize(text_prompts).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Get top prediction
    probs = similarity[0].cpu().numpy()
    top_idx = probs.argmax()

    return categories[top_idx], probs[top_idx]

# Example
categories = ["cat", "dog", "car", "airplane", "tree"]
label, confidence = classify_image("image.jpg", categories)
print(f"Predicted: {label} ({confidence:.2%})")

# Works with ANY categories - no retraining needed!
'''

    print(code)


def text_to_image_search():
    """Show text-to-image search."""
    print("\n" + "=" * 70)
    print("=== TEXT-TO-IMAGE SEARCH ===\n")

    code = '''def search_images(query, image_paths, top_k=5):
    """Find images most similar to text query."""

    # Encode query
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Encode all images
    image_features = []
    for img_path in image_paths:
        image = Image.open(img_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
            image_features.append(features)

    image_features = torch.cat(image_features)

    # Compute similarities
    similarities = (text_features @ image_features.T)[0]

    # Get top K
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "path": image_paths[idx],
            "score": similarities[idx].item()
        })

    return results

# Example
query = "a sunset over the ocean"
results = search_images(query, all_image_paths, top_k=5)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['path']}: {result['score']:.3f}")
'''

    print(code)


def clip_models():
    """Compare CLIP model variants."""
    print("\n" + "=" * 70)
    print("=== CLIP MODEL VARIANTS ===\n")

    print("Official OpenAI models:\n")

    models = [
        ("ViT-B/32", "Fast", "224x224", "Balanced speed/quality"),
        ("ViT-B/16", "Medium", "224x224", "Better quality"),
        ("ViT-L/14", "Slow", "224x224", "Best quality"),
        ("RN50", "Fast", "224x224", "ResNet-based, faster"),
        ("RN101", "Medium", "224x224", "Deeper ResNet"),
    ]

    print("Model       Speed     Input Size   Use Case")
    print("-" * 60)
    for name, speed, size, use_case in models:
        print(f"{name:12}{speed:10}{size:13}{use_case}")

    print("\n\nOpen source alternatives:")
    print("  • OpenCLIP: Open source implementation with more models")
    print("  • CLIP-ViT-g-14: Larger model, better performance")
    print("  • LaCLIP: Improved with additional data")


def prompt_engineering_for_clip():
    """Show prompt engineering techniques."""
    print("\n" + "=" * 70)
    print("=== PROMPT ENGINEERING FOR CLIP ===\n")

    print("1. Template prompts")
    print("   Improves classification accuracy\n")

    code1 = '''# Instead of just "cat"
text_prompts = [f"a photo of a {category}" for category in categories]

# Even better:
text_prompts = [
    f"a photo of a {category}, a type of animal"
    for category in animal_categories
]
'''
    print(code1)

    print("\n2. Multiple descriptions")
    print("   Average embeddings for robustness\n")

    code2 = '''templates = [
    "a photo of a {}",
    "an image of a {}",
    "a picture showing a {}"
]

# Encode all variations
embeddings = []
for template in templates:
    text = template.format(category)
    embedding = encode_text(text)
    embeddings.append(embedding)

# Average
category_embedding = torch.stack(embeddings).mean(dim=0)
'''
    print(code2)

    print("\n3. Descriptive queries")
    print("   More specific = better results\n")

    print("   ✗ 'dog'")
    print("   ✓ 'a golden retriever dog in a park'")
    print("   ✓ 'a small brown dog wearing a red collar'")


def performance_optimization():
    """Show performance optimization."""
    print("\n" + "=" * 70)
    print("=== PERFORMANCE OPTIMIZATION ===\n")

    code = '''# 1. Batch processing
images = [preprocess(Image.open(p)).unsqueeze(0) for p in paths]
image_batch = torch.cat(images).to(device)

with torch.no_grad():
    features = model.encode_image(image_batch)  # Process all at once

# 2. Caching embeddings
import numpy as np

# Compute once
image_embeddings = {}
for path in image_paths:
    emb = encode_image(path)
    image_embeddings[path] = emb.cpu().numpy()

# Save
np.savez("embeddings.npz", **image_embeddings)

# Load for quick search
embeddings = np.load("embeddings.npz")

# 3. GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 4. Half precision
model = model.half()  # FP16 for faster inference
'''

    print(code)


def limitations():
    """Discuss limitations."""
    print("\n" + "=" * 70)
    print("=== LIMITATIONS ===\n")

    print("Challenges:")
    print("  • Limited understanding of fine-grained details")
    print("  • Struggles with uncommon objects/concepts")
    print("  • Cannot count objects accurately")
    print("  • Biases from training data")
    print("  • Fixed input size (crops/resizes images)")
    print("  • No spatial understanding (where in image)\n")

    print("When to use alternatives:")
    print("  → Precise counting: Use object detection (YOLO, Faster R-CNN)")
    print("  → Fine details: Use task-specific models")
    print("  → Complex reasoning: Use vision-language models (GPT-4V)")
    print("  → Domain-specific: Fine-tune CLIP or use specialized models")


def real_world_example():
    """Show real-world example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE ===\n")

    print("Task: Content moderation system\n")

    code = '''def moderate_image(image_path):
    """Check if image contains inappropriate content."""

    categories = [
        "safe content",
        "violence",
        "adult content",
        "hate symbols",
        "graphic content"
    ]

    label, confidence = classify_image(image_path, categories)

    if label == "safe content" and confidence > 0.7:
        return {"approved": True, "reason": "safe"}
    else:
        return {
            "approved": False,
            "reason": label,
            "confidence": confidence,
            "review_required": confidence < 0.9
        }

# Benefits:
# - Zero-shot (no training data needed)
# - Fast (< 100ms per image)
# - Customizable categories
# - Can add new categories instantly
'''

    print(code)


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Use templates", "Wrap categories in 'a photo of a {}'"),
        ("Batch processing", "Process multiple images at once"),
        ("Cache embeddings", "Compute once, search many times"),
        ("Normalize features", "Always normalize for cosine similarity"),
        ("Multiple descriptions", "Average embeddings for robustness"),
        ("Appropriate model", "ViT-B/32 for speed, ViT-L/14 for quality"),
        ("GPU if available", "10-100× faster than CPU"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


if __name__ == "__main__":
    clip_intro()
    basic_usage()
    zero_shot_classification()
    text_to_image_search()
    clip_models()
    prompt_engineering_for_clip()
    performance_optimization()
    limitations()
    real_world_example()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: CLIP enables zero-shot image understanding!")
    print("No training data needed - just describe what you're looking for")

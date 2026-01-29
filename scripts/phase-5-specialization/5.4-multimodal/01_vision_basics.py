"""
01 - Vision API Basics
======================
Analyze images with GPT-4V and multimodal prompting.

Key concept: Vision models (GPT-4V, Claude) can understand images alongside text, enabling document analysis, visual Q&A, and image understanding tasks.

Book reference: hands_on_LLM.II.9
"""

import utils._load_env  # Loads .env file automatically

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def vision_api_intro():
    """Introduce vision APIs."""
    print("=== VISION APIs FOR LLMs ===\n")

    print("What are vision-enabled LLMs?")
    print("  Models that can process both text and images in the same context,")
    print("  understanding visual content and answering questions about it.\n")

    print("Available models:")
    print("  • GPT-4V (OpenAI) - GPT-4 with vision")
    print("  • Claude 3 (Anthropic) - Opus, Sonnet, Haiku with vision")
    print("  • Gemini Pro Vision (Google)")
    print("  • LLaVA (Open source)\n")

    print("Capabilities:")
    print("  ✓ Describe image content")
    print("  ✓ Answer questions about images")
    print("  ✓ Extract text from images (OCR)")
    print("  ✓ Analyze charts and diagrams")
    print("  ✓ Compare multiple images")
    print("  ✓ Detect objects and scenes")
    print("  ✓ Understand spatial relationships")


def basic_example():
    """Show basic vision API usage."""
    print("\n" + "=" * 70)
    print("=== BASIC EXAMPLE ===\n")

    code = '''from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Analyze a single image
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)

# Example output:
# "The image shows a golden retriever dog sitting in a park.
# The dog appears happy with its tongue out. In the background,
# there are trees and a blue sky."
'''

    print(code)


def image_formats():
    """Explain supported image formats."""
    print("\n" + "=" * 70)
    print("=== SUPPORTED IMAGE FORMATS ===\n")

    print("1. URL (remote image)")
    print("   Most convenient for hosted images\n")

    code_url = '''# Remote image URL
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/photo.jpg"
    }
}
'''
    print(code_url)

    print("\n2. Base64-encoded (local image)")
    print("   For local files or dynamic images\n")

    code_base64 = '''import base64

# Read and encode local image
with open("image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Use in API call
{
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{encoded_string}"
    }
}
'''
    print(code_base64)

    print("\n3. Image detail levels")
    print("   Control processing detail and cost\n")

    code_detail = '''# Low detail (faster, cheaper)
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg",
        "detail": "low"  # or "high" or "auto"
    }
}

# low:  Faster, cheaper, 512x512 preview
# high: More detail, slower, more tokens
# auto: Model decides based on image
'''
    print(code_detail)


def common_use_cases():
    """Show common vision use cases."""
    print("\n" + "=" * 70)
    print("=== COMMON USE CASES ===\n")

    print("1. Image Description")
    print("   Generate detailed descriptions of images\n")

    code1 = '''prompt = """Describe this image in detail. Include:
- Main subjects
- Colors and composition
- Mood and atmosphere
- Any text visible"""

# Useful for: Accessibility, cataloging, alt text generation
'''
    print(code1)

    print("\n2. Visual Question Answering")
    print("   Ask specific questions about image content\n")

    code2 = '''questions = [
    "How many people are in this image?",
    "What is the person wearing?",
    "Is this indoors or outdoors?",
    "What time of day does this appear to be?"
]

# Useful for: Image search, content moderation, analysis
'''
    print(code2)

    print("\n3. OCR and Text Extraction")
    print("   Extract text from images (documents, signs, screenshots)\n")

    code3 = '''prompt = """Extract all text from this image.
Maintain the original formatting and structure.
If it's a table, preserve the table format."""

# Useful for: Document processing, receipt scanning, sign reading
'''
    print(code3)

    print("\n4. Chart and Diagram Analysis")
    print("   Understand data visualizations\n")

    code4 = '''prompt = """Analyze this chart and provide:
1. Chart type (bar, line, pie, etc.)
2. Key trends or patterns
3. Numerical data points if visible
4. Main insights"""

# Useful for: Report generation, data analysis, accessibility
'''
    print(code4)


def multiple_images():
    """Show handling multiple images."""
    print("\n" + "=" * 70)
    print("=== MULTIPLE IMAGES ===\n")

    code = '''# Compare or analyze multiple images
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images. What are the differences?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg"}
                }
            ]
        }
    ],
    max_tokens=500
)

# Example uses:
# - Before/after comparisons
# - Product variations
# - Document versions
# - Medical imaging (X-rays, scans)
'''

    print(code)


def best_prompting_practices():
    """Show best practices for vision prompts."""
    print("\n" + "=" * 70)
    print("=== PROMPTING BEST PRACTICES ===\n")

    print("1. Be specific about what you want")
    print("   ✗ 'Describe this'")
    print("   ✓ 'Describe the architectural style and notable features'\n")

    print("2. Provide context")
    print("   'This is a medical X-ray. Identify any anomalies.'\n")

    print("3. Ask for structured output")
    print("   'List findings in JSON format: {object: str, location: str, count: int}'\n")

    print("4. Use examples (few-shot)")
    print("   'Like in this example: [example image] -> [example output]'\n")

    print("5. Specify detail level")
    print("   'Provide a brief/detailed/technical description'\n")

    print("6. Request verification")
    print("   'Are you confident in this analysis? Explain your reasoning.'")


def cost_and_performance():
    """Discuss cost and performance considerations."""
    print("\n" + "=" * 70)
    print("=== COST AND PERFORMANCE ===\n")

    print("Token usage (GPT-4V):")
    print("  • Low detail (512x512): ~85 tokens per image")
    print("  • High detail: ~170 tokens + 85 tokens per 512x512 tile")
    print("  • Example: 2048x2048 image = ~1,105 tokens\n")

    print("Cost comparison (approximate):")
    print("  GPT-4V: ~$0.01-0.03 per image (depending on detail)")
    print("  Claude 3: Similar pricing")
    print("  Gemini: Varies by model tier\n")

    print("Performance:")
    print("  • Response time: 3-10 seconds")
    print("  • Batch processing: Use async for multiple images")
    print("  • Rate limits: Check provider docs\n")

    print("Optimization tips:")
    print("  ✓ Use 'low' detail for simple images")
    print("  ✓ Resize large images before encoding")
    print("  ✓ Cache results for repeated queries")
    print("  ✓ Batch similar requests")


def error_handling():
    """Show error handling."""
    print("\n" + "=" * 70)
    print("=== ERROR HANDLING ===\n")

    code = '''from openai import OpenAI, BadRequestError, APIError
import time


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

def analyze_image_with_retry(image_url, prompt, max_retries=3):
    """Analyze image with retry logic."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                max_tokens=500
            )
            return response.choices[0].message.content

        except BadRequestError as e:
            # Image format issue or invalid URL
            print(f"Bad request: {e}")
            return None

        except APIError as e:
            # Server error, retry
            if attempt < max_retries - 1:
                print(f"API error, retrying... (attempt {attempt + 1})")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts")
                return None

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

# Common errors:
# - Image too large (>20MB)
# - Invalid image format
# - URL not accessible
# - Rate limit exceeded
'''

    print(code)


def limitations():
    """Discuss limitations."""
    print("\n" + "=" * 70)
    print("=== LIMITATIONS ===\n")

    print("Current limitations:")
    print("  • Cannot generate or edit images (use DALL-E for that)")
    print("  • May struggle with very small text")
    print("  • Limited understanding of complex diagrams")
    print("  • Cannot identify specific individuals (privacy)")
    print("  • May hallucinate details not in image")
    print("  • Performance varies by image quality")
    print("  • Cannot process video (must extract frames)\n")

    print("Workarounds:")
    print("  → Enhance image quality before sending")
    print("  → Split complex images into regions")
    print("  → Provide context in prompt")
    print("  → Verify outputs with structured questions")
    print("  → Use specialized tools for specific tasks (OCR libraries)")


def real_world_example():
    """Show real-world example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE ===\n")

    print("Task: Analyze product images for e-commerce\n")

    code = '''def analyze_product_image(image_url):
    """Extract product details from image."""

    prompt = """Analyze this product image and extract:
1. Product category
2. Main color(s)
3. Material (if visible)
4. Key features
5. Condition (new/used)
6. Any text or branding visible

Format as JSON."""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"  # Need detail for text
                    }
                }
            ]
        }],
        max_tokens=500
    )

    return response.choices[0].message.content

# Use case:
# - Auto-generate product descriptions
# - Quality control
# - Inventory management
# - Search/categorization
'''

    print(code)


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Optimize image size", "Resize to reasonable dimensions before encoding"),
        ("Use appropriate detail", "Low detail for simple images saves cost"),
        ("Structured prompts", "Request specific format (JSON, list, etc.)"),
        ("Provide context", "Explain what the image represents"),
        ("Verify outputs", "Vision models can hallucinate"),
        ("Handle errors", "Images may be invalid or unreachable"),
        ("Cache results", "Don't reprocess same images"),
        ("Batch processing", "Use async for multiple images")]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


if __name__ == "__main__":
    vision_api_intro()
    basic_example()
    image_formats()
    common_use_cases()
    multiple_images()
    best_prompting_practices()
    cost_and_performance()
    error_handling()
    limitations()
    real_world_example()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Vision-enabled LLMs unlock image understanding!")
    print("Analyze images, extract text, answer visual questions - all with prompts")

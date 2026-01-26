# Module 5.4: Multimodal

> *"Work with images and text together - unlock visual understanding"*

This module covers multimodal AI - analyzing images with GPT-4V, text-image similarity with CLIP, building CLIP-based search systems, and extracting structured information from document images.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_vision_basics.py` | Vision API Basics | Analyze images with GPT-4V and multimodal prompting |
| `02_clip_basics.py` | CLIP Basics | Text-image similarity with CLIP embeddings |
| `03_image_text_search.py` | Image-Text Search | Build CLIP-based multimodal search system |
| `04_document_vision.py` | Document Vision | Extract structured data from document images |

## Why Multimodal?

Text-only AI is limited. Multimodal AI enables:
- **Visual Understanding**: Analyze images, not just text
- **Document Processing**: Extract from receipts, forms, invoices
- **Semantic Search**: Find images with natural language
- **Richer Applications**: Combine text and visual information
- **Better UX**: Users can search/interact with images

## Core Concepts

### 1. Vision-Language Models (GPT-4V, Claude)

Process text and images together:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
)
```

**Capabilities:**
- Describe image content
- Answer questions about images
- Extract text (OCR)
- Analyze charts/diagrams
- Compare multiple images

### 2. CLIP (Contrastive Language-Image Pre-training)

Unified embedding space for text and images:

```python
import clip

model, preprocess = clip.load("ViT-B/32")

# Encode image
image_features = model.encode_image(image)

# Encode text
text_features = model.encode_text(text)

# Similarity
similarity = (image_features @ text_features.T)
```

**Zero-shot classification:**
- No training data needed
- Just describe categories
- Works with any concepts

### 3. Multimodal Search

Search across modalities:

```
Text → Image:    "sunset over ocean" → [beach photos]
Image → Image:   [cat photo] → [similar cats]
Image → Text:    [product] → "blue running shoes"
```

### 4. Document Vision

Extract structured data from documents:

```python
prompt = """Extract from this invoice:
{
  "invoice_number": "",
  "date": "",
  "total": 0,
  "items": []
}
"""

data = extract_from_document("invoice.jpg", prompt)
```

**Use cases:**
- Receipt processing
- Form extraction
- ID card reading
- Contract analysis

## Running the Examples

Each script demonstrates different multimodal capabilities:

```bash
# Vision API basics
python modules/phase5/5.4-multimodal/01_vision_basics.py

# CLIP embeddings
python modules/phase5/5.4-multimodal/02_clip_basics.py

# Build image search
python modules/phase5/5.4-multimodal/03_image_text_search.py

# Document extraction
python modules/phase5/5.4-multimodal/04_document_vision.py
```

## Practical Workflows

### Workflow 1: Image Analysis with GPT-4V

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_image(image_path, question):
    """Ask questions about an image."""

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }]
    )

    return response.choices[0].message.content

# Use it
answer = analyze_image("photo.jpg", "How many people are in this image?")
```

### Workflow 2: Build CLIP Search System

```python
import clip
import torch

class ImageSearch:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.image_embeddings = []
        self.image_paths = []

    def index_images(self, paths):
        """Index images for search."""
        for path in paths:
            image = Image.open(path)
            image_input = self.preprocess(image).unsqueeze(0)

            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)

            self.image_embeddings.append(features)
            self.image_paths.append(path)

    def search(self, query, top_k=5):
        """Search images with text."""
        text_input = clip.tokenize([query])

        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        image_embs = torch.cat(self.image_embeddings)
        similarities = (image_embs @ text_features.T).squeeze()

        # Top K
        top_indices = similarities.argsort(descending=True)[:top_k]

        return [(self.image_paths[i], similarities[i].item())
                for i in top_indices]

# Use it
search = ImageSearch()
search.index_images(image_paths)
results = search.search("a dog in a park")
```

### Workflow 3: Document Extraction

```python
def extract_invoice(image_path):
    """Extract structured data from invoice."""

    prompt = """Extract invoice information as JSON:
{
  "invoice_number": "",
  "date": "",
  "vendor": "",
  "total": 0.00,
  "items": [{"name": "", "price": 0}]
}
"""

    response = analyze_image(image_path, prompt)

    # Parse JSON
    import json
    try:
        data = json.loads(response)
        return data
    except:
        return None

# Process invoices
data = extract_invoice("invoice.jpg")
if data:
    print(f"Invoice #{data['invoice_number']}: ${data['total']}")
```

## Best Practices

### 1. Vision APIs (GPT-4V)
- **Use high detail for documents**: Text needs clarity
- **Be specific in prompts**: "Extract table as JSON" vs "What's this?"
- **Request structured output**: Ask for JSON format
- **Handle errors**: Image may be invalid or API may fail
- **Optimize images**: Resize large images before encoding
- **Cache results**: Don't reprocess same images

### 2. CLIP
- **Use templates**: "a photo of a {category}" improves classification
- **Normalize embeddings**: Always normalize for cosine similarity
- **Batch processing**: Process multiple images at once
- **Choose right model**: ViT-B/32 (fast) vs ViT-L/14 (accurate)
- **Cache embeddings**: Compute once, search many times
- **GPU if available**: 10-100× faster

### 3. Image Search
- **Use FAISS for scale**: Essential for >10K images
- **Hybrid search**: Combine semantic + metadata filters
- **Monitor quality**: Track MRR, Recall@K
- **Update incrementally**: Don't rebuild entire index
- **Appropriate similarity threshold**: Filter low-confidence results

### 4. Document Vision
- **High quality images**: Minimum 300 DPI for documents
- **Structured prompts**: Provide exact JSON schema
- **Validate outputs**: Parse and verify extracted data
- **Two-pass extraction**: Extract then structure
- **Human review queue**: Flag low-confidence extractions

## Common Pitfalls

### 1. Wrong Image Format
**Problem**: Invalid images or unsupported formats
**Solution**: Validate and convert images before processing

### 2. Not Normalizing CLIP Embeddings
**Problem**: Incorrect similarity scores
**Solution**: Always normalize embeddings for cosine similarity

### 3. Low Image Quality
**Problem**: Poor extraction from blurry/low-res images
**Solution**: Enhance contrast, increase DPI, crop/deskew

### 4. Expecting Perfect Accuracy
**Problem**: Vision models can hallucinate or miss details
**Solution**: Validate outputs, use confidence scores, human review

### 5. Ignoring Costs
**Problem**: Expensive at scale ($0.01-0.03 per image)
**Solution**: Optimize images, cache results, use low detail when possible

## Tools and Libraries

### Vision APIs
```bash
# OpenAI GPT-4V
pip install openai

# Anthropic Claude with vision
pip install anthropic

# Google Gemini
pip install google-generativeai
```

### CLIP
```bash
# Official CLIP
pip install git+https://github.com/openai/CLIP.git

# Open CLIP (more models)
pip install open-clip-torch

# For image processing
pip install Pillow torch torchvision
```

### Vector Search
```bash
# FAISS (fast similarity search)
pip install faiss-cpu  # or faiss-gpu

# Alternative vector databases
pip install chromadb  # Simple vector DB
pip install qdrant-client  # Production-ready
pip install weaviate-client  # Open source
```

## Model Comparison

### Vision APIs
```
Model          Quality    Speed    Cost/Image    Use Case
─────────────────────────────────────────────────────────
GPT-4V         Excellent  5-10s    $0.01-0.03    General vision
Claude 3       Excellent  5-10s    Similar       Documents
Gemini Pro     Very Good  3-8s     Lower         Cost-sensitive
LLaVA (local)  Good       1-3s     Free          Privacy
```

### CLIP Models
```
Model          Speed    Accuracy   Size    Use Case
───────────────────────────────────────────────────
ViT-B/32       Fast     Good       350MB   Production
ViT-B/16       Medium   Better     350MB   Balanced
ViT-L/14       Slow     Best       890MB   Quality priority
RN50           Fastest  Good       380MB   Speed priority
```

## Real-World Examples

### Example 1: E-commerce Product Search
```
Problem: Help users find products with descriptions
Solution: CLIP-based semantic search
Data: 100K product images
Results:
  - Search latency: 45ms
  - Relevance: +35% vs keyword search
  - Conversion: +12%
```

### Example 2: Invoice Processing
```
Problem: Automate invoice data entry
Solution: GPT-4V document extraction
Volume: 10K invoices/month
Results:
  - Accuracy: 94%
  - Manual review: Only 6%
  - Time saved: 80%
```

### Example 3: Medical Image Analysis
```
Problem: Triage X-rays for review
Solution: CLIP classification + GPT-4V analysis
Data: 50K X-rays
Results:
  - Triage accuracy: 89%
  - Radiologist time: -40%
  - Earlier detection: +15%
```

## Cost Optimization

### Vision API Costs
```python
# Reduce image size
from PIL import Image

def optimize_image(path, max_size=2048):
    img = Image.open(path)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(d * ratio) for d in img.size)
        img = img.resize(new_size)
    return img

# Use 'low' detail for simple images
{
    "type": "image_url",
    "image_url": {
        "url": "...",
        "detail": "low"  # Cheaper
    }
}

# Typical savings: 50-70%
```

### CLIP: Free but Hardware
```
GPU Needed:
  Small models (ViT-B/32): Any GPU with 4GB VRAM
  Large models (ViT-L/14): GPU with 8GB+ VRAM

Cloud costs:
  AWS p3.2xlarge: $3.06/hour
  Process 10K images/hour: $0.0003/image

vs OpenAI GPT-4V: $0.01-0.03/image
```

## Book References

- `hands_on_LLM.II.9` - Multimodal LLMs
- `AI_eng.6` - Document processing
- `AI_eng.9` - Vision applications

## Next Steps

After mastering multimodal:
- Module 5.1: Fine-tuning LLMs - Fine-tune vision models
- Module 5.2: Custom Embeddings - Combine with CLIP embeddings
- Module 3.4: Advanced RAG - Multimodal RAG systems
- Module 4.7: Cloud Deployment - Deploy vision systems

## Additional Resources

- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **GPT-4V System Card**: https://openai.com/research/gpt-4v-system-card
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **FAISS**: https://github.com/facebookresearch/faiss
- **LLaVA**: https://llava-vl.github.io/

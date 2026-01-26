"""
04 - Document Vision
=====================
Extract information from document images using vision models.

Key concept: Vision-enabled LLMs can extract structured information from documents, forms, receipts, and scanned papers without traditional OCR pipelines.

Book reference: hands_on_LLM.II.9, AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def document_vision_intro():
    """Introduce document vision."""
    print("=== DOCUMENT VISION ===\n")

    print("What is document vision?")
    print("  Using vision-enabled LLMs to extract structured information from")
    print("  document images (forms, receipts, invoices, contracts, etc.)\n")

    print("Advantages over traditional OCR:")
    print("  ✓ Understands context, not just text")
    print("  ✓ Handles complex layouts")
    print("  ✓ No pre-processing required")
    print("  ✓ Extracts structured data directly")
    print("  ✓ Works with handwriting")
    print("  ✓ Understands tables and forms\n")

    print("Use cases:")
    print("  • Invoice/receipt processing")
    print("  • Form extraction")
    print("  • ID card/passport reading")
    print("  • Contract analysis")
    print("  • Medical record extraction")
    print("  • Insurance claims processing")


def basic_document_extraction():
    """Show basic document extraction."""
    print("\n" + "=" * 70)
    print("=== BASIC DOCUMENT EXTRACTION ===\n")

    code = '''from openai import OpenAI
import base64

client = OpenAI()

def extract_from_document(image_path, prompt):
    """Extract information from document image."""

    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # Important for text clarity
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

# Example: Receipt extraction
prompt = """Extract all information from this receipt:
- Store name
- Date
- Items (name, quantity, price)
- Subtotal
- Tax
- Total

Format as JSON."""

receipt_data = extract_from_document("receipt.jpg", prompt)
print(receipt_data)
'''

    print(code)


def structured_extraction_examples():
    """Show structured extraction for different document types."""
    print("\n" + "=" * 70)
    print("=== STRUCTURED EXTRACTION EXAMPLES ===\n")

    print("1. Invoice Processing\n")

    code1 = '''def extract_invoice(image_path):
    """Extract invoice data."""

    prompt = """Extract the following from this invoice:
{
  "invoice_number": "",
  "date": "",
  "vendor": {
    "name": "",
    "address": ""
  },
  "customer": {
    "name": "",
    "address": ""
  },
  "line_items": [
    {
      "description": "",
      "quantity": 0,
      "unit_price": 0,
      "total": 0
    }
  ],
  "subtotal": 0,
  "tax": 0,
  "total": 0
}

Return only valid JSON."""

    return extract_from_document(image_path, prompt)
'''
    print(code1)

    print("\n2. ID Card/Passport\n")

    code2 = '''def extract_id_card(image_path):
    """Extract ID card information."""

    prompt = """Extract personal information:
- Full name
- Date of birth
- ID number
- Issue date
- Expiry date
- Address (if visible)

Format as JSON. If any field is not visible, use null."""

    return extract_from_document(image_path, prompt)
'''
    print(code2)

    print("\n3. Form Extraction\n")

    code3 = '''def extract_form(image_path, field_definitions):
    """Extract form fields."""

    fields_list = "\\n".join([f"- {name}: {desc}" for name, desc in field_definitions.items()])

    prompt = f"""Extract the following fields from this form:
{fields_list}

Return as JSON with exact field names as keys."""

    return extract_from_document(image_path, prompt)

# Example
fields = {
    "applicant_name": "Full name",
    "date_of_birth": "DOB",
    "phone_number": "Contact number",
    "email": "Email address"
}

data = extract_form("application.jpg", fields)
'''
    print(code3)


def table_extraction():
    """Show table extraction."""
    print("\n" + "=" * 70)
    print("=== TABLE EXTRACTION ===\n")

    code = '''def extract_table(image_path):
    """Extract table data from document."""

    prompt = """This image contains a table. Extract it as:
1. Identify all columns
2. Extract all rows
3. Format as JSON array of objects

Example format:
[
  {"column1": "value", "column2": "value"},
  {"column1": "value", "column2": "value"}
]

Preserve data types (numbers as numbers, dates as strings)."""

    response = extract_from_document(image_path, prompt)

    # Parse JSON
    import json
    try:
        table_data = json.loads(response)
        return table_data
    except json.JSONDecodeError:
        # If model returns text explanation, try again with stricter prompt
        return None

# Example: Financial statement table
table = extract_table("financial_statement.png")
if table:
    for row in table:
        print(row)
'''

    print(code)


def multi_page_documents():
    """Show multi-page document handling."""
    print("\n" + "=" * 70)
    print("=== MULTI-PAGE DOCUMENTS ===\n")

    code = '''def extract_from_pdf(pdf_path, prompt):
    """Extract from multi-page PDF."""

    # Convert PDF to images
    from pdf2image import convert_from_path

    pages = convert_from_path(pdf_path)

    # Process each page
    all_data = []

    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}")

        # Save page as temporary image
        temp_path = f"temp_page_{i}.jpg"
        page.save(temp_path, "JPEG")

        # Extract
        data = extract_from_document(temp_path, prompt)
        all_data.append({
            "page": i + 1,
            "content": data
        })

    return all_data

# Example: Multi-page contract
contract_data = extract_from_pdf("contract.pdf", """
Extract key clauses:
- Party names
- Effective date
- Terms and conditions
- Payment terms
- Termination clauses
""")
'''

    print(code)


def validation_and_error_handling():
    """Show validation patterns."""
    print("\n" + "=" * 70)
    print("=== VALIDATION & ERROR HANDLING ===\n")

    code = '''import json
from typing import Optional, Dict, Any

def extract_with_validation(image_path, schema: Dict[str, Any]) -> Optional[Dict]:
    """Extract with schema validation."""

    # Build prompt from schema
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""Extract information matching this schema:
{schema_str}

Rules:
1. Return ONLY valid JSON
2. Use null for missing fields
3. Match data types exactly
4. Validate numbers and dates

Return JSON:"""

    # Extract
    response = extract_from_document(image_path, prompt)

    # Parse and validate
    try:
        data = json.loads(response)

        # Basic schema validation
        if not validate_schema(data, schema):
            print("Warning: Data doesn't match schema")
            return None

        return data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None

def validate_schema(data: Dict, schema: Dict) -> bool:
    """Simple schema validation."""
    for key, expected_type in schema.items():
        if key not in data:
            return False

        value = data[key]
        if value is not None:
            # Check type (simplified)
            if expected_type == "str" and not isinstance(value, str):
                return False
            if expected_type == "int" and not isinstance(value, (int, float)):
                return False

    return True

# Example usage
schema = {
    "invoice_number": "str",
    "total": "float",
    "date": "str"
}

data = extract_with_validation("invoice.jpg", schema)
if data:
    print("Valid extraction:", data)
else:
    print("Extraction failed or invalid")
'''

    print(code)


def quality_improvement_tips():
    """Show tips for improving extraction quality."""
    print("\n" + "=" * 70)
    print("=== QUALITY IMPROVEMENT TIPS ===\n")

    print("1. Image quality")
    print("   • Minimum 300 DPI for documents")
    print("   • Good contrast and lighting")
    print("   • No blur or distortion\n")

    print("2. Prompt engineering")
    print("   • Be very specific about format")
    print("   • Provide example JSON structure")
    print("   • Request validation and error checking")
    print("   • Ask for confidence scores\n")

    print("3. Two-pass extraction")

    code = '''def two_pass_extraction(image_path):
    """First pass: extract, second pass: validate."""

    # Pass 1: Extract
    prompt1 = "Extract all text and structure from this document."
    raw_text = extract_from_document(image_path, prompt1)

    # Pass 2: Structure
    prompt2 = f"""Given this extracted text:
{raw_text}

Format as structured JSON with fields:
- field1
- field2
...
"""

    structured = extract_from_document(image_path, prompt2)
    return structured
'''
    print(code)

    print("\n4. Hybrid approach")
    print("   Combine vision model with traditional OCR for verification\n")

    code2 = '''from pytesseract import image_to_string

def hybrid_extraction(image_path):
    """Combine vision LLM with OCR."""

    # Method 1: Vision LLM
    vision_result = extract_from_document(image_path, "Extract all text")

    # Method 2: Traditional OCR
    ocr_text = image_to_string(Image.open(image_path))

    # Compare and merge
    # Use vision LLM for structure, OCR for verification
    # ...
'''
    print(code2)


def cost_optimization():
    """Show cost optimization strategies."""
    print("\n" + "=" * 70)
    print("=== COST OPTIMIZATION ===\n")

    print("Strategies:")
    print("  1. Reduce image size (maintain readability)")
    print("  2. Use 'low' detail for simple documents")
    print("  3. Batch process multiple documents")
    print("  4. Cache results for similar documents")
    print("  5. Pre-process: crop, deskew, enhance contrast\n")

    code = '''from PIL import Image

def optimize_document_image(image_path, max_size=2048):
    """Optimize image for document extraction."""

    img = Image.open(image_path)

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Enhance contrast for text
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    # Save optimized
    output_path = "optimized_" + image_path
    img.save(output_path, quality=85, optimize=True)

    return output_path

# Typical savings: 50-70% reduction in API costs
'''

    print(code)


def real_world_example():
    """Show real-world example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE: EXPENSE AUTOMATION ===\n")

    print("System: Automated expense report processing\n")

    code = '''class ExpenseProcessor:
    """Process expense receipts automatically."""

    def __init__(self):
        self.client = OpenAI()

    def process_receipt(self, receipt_image):
        """Extract expense data from receipt."""

        prompt = """Extract expense information:
{
  "merchant": "",
  "date": "YYYY-MM-DD",
  "amount": 0.00,
  "currency": "",
  "category": "meals|travel|supplies|other",
  "items": [
    {"name": "", "price": 0.00}
  ],
  "payment_method": "cash|card"
}

If uncertain, mark field as null."""

        response = extract_from_document(receipt_image, prompt)

        try:
            data = json.loads(response)

            # Validate
            if self.validate_expense(data):
                return {"status": "approved", "data": data}
            else:
                return {"status": "review_needed", "data": data}

        except json.JSONDecodeError:
            return {"status": "failed", "error": "Could not parse"}

    def validate_expense(self, data):
        """Business logic validation."""
        # Check amount
        if data.get("amount", 0) > 1000:
            return False  # Requires manager approval

        # Check required fields
        required = ["merchant", "date", "amount"]
        if not all(data.get(f) for f in required):
            return False

        return True

# Results:
# - Processing time: 3 seconds per receipt
# - Accuracy: 94%
# - Manual review: Only 6% of receipts
# - Time saved: 80% vs manual entry
'''

    print(code)


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("High detail for documents", "Use 'high' detail setting for text clarity"),
        ("Structured prompts", "Request specific JSON schema"),
        ("Validate outputs", "Parse and verify extracted data"),
        ("Optimize images", "Enhance contrast, crop, deskew"),
        ("Two-pass extraction", "Extract then structure for better results"),
        ("Cache results", "Don't reprocess same documents"),
        ("Error handling", "Handle OCR failures gracefully"),
        ("Human review queue", "Flag low-confidence extractions"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


if __name__ == "__main__":
    document_vision_intro()
    basic_document_extraction()
    structured_extraction_examples()
    table_extraction()
    multi_page_documents()
    validation_and_error_handling()
    quality_improvement_tips()
    cost_optimization()
    real_world_example()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Vision LLMs revolutionize document processing!")
    print("Extract structured data from any document with just prompts")

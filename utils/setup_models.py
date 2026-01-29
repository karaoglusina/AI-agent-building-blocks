#!/usr/bin/env python3
"""Download required spaCy and NLTK models/data."""

import subprocess
import sys

def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False

def main():
    """Download all required models and data."""
    print("Setting up NLP models and data...")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Download spaCy model
    total_count += 1
    if run_command(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        "Downloading spaCy English model (en_core_web_sm)"
    ):
        success_count += 1
    
    # Download NLTK data
    nltk_data = [
        ("punkt", "Punkt tokenizer"),
        ("stopwords", "Stopwords corpus"),
        ("averaged_perceptron_tagger", "POS tagger"),
        ("wordnet", "WordNet corpus"),
    ]
    
    for data_name, description in nltk_data:
        total_count += 1
        # Use Python to download NLTK data
        download_script = f"""
import nltk
try:
    nltk.download('{data_name}', quiet=True)
    print("Success")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
"""
        if run_command(
            [sys.executable, "-c", download_script],
            f"Downloading NLTK {description}"
        ):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Setup complete: {success_count}/{total_count} items downloaded successfully")
    
    if success_count < total_count:
        print("\nSome downloads failed. You may need to install them manually:")
        print("  python -m spacy download en_core_web_sm")
        print("  python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        sys.exit(1)

if __name__ == "__main__":
    main()

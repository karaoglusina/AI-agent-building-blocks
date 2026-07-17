#!/usr/bin/env python3
"""Download the spaCy model and NLTK data the Phase 2/3/5 NLP scripts need.

Run once after `uv sync`. Note that `uv sync` can uninstall the spaCy model (it isn't
a tracked dependency), so re-run this if spaCy suddenly can't find `en_core_web_sm`.

Each NLTK resource is verified with `nltk.data.find` after downloading, because
`nltk.download(quiet=True)` returns False on failure rather than raising — checking
the return value alone will happily report success for data that never arrived.
"""

import subprocess
import sys

# (download id, path passed to nltk.data.find, human description)
# punkt_tab / averaged_perceptron_tagger_eng are the NLTK >=3.9 replacements for
# punkt / averaged_perceptron_tagger. Both are downloaded: scripts reference each.
NLTK_RESOURCES = [
    ("punkt", "tokenizers/punkt", "Punkt tokenizer"),
    ("punkt_tab", "tokenizers/punkt_tab", "Punkt tokenizer tables (NLTK >=3.9)"),
    ("stopwords", "corpora/stopwords", "Stopwords corpus"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger", "POS tagger"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng", "POS tagger (NLTK >=3.9)"),
    ("wordnet", "corpora/wordnet", "WordNet corpus"),
    ("opinion_lexicon", "corpora/opinion_lexicon", "Opinion lexicon"),
]


def download_spacy_model() -> bool:
    """Download the spaCy English model, unless uv already installed it.

    The phase-2/3/4 dependency groups declare `en-core-web-sm` (see [tool.uv.sources]),
    so `uv sync` normally provides it. Downloading over a working install can leave a
    half-written package directory, so check before touching it.
    """
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("\n✓ spaCy en_core_web_sm already installed")
        return True
    except Exception:  # noqa: BLE001 - not installed, or installed broken; reinstall
        pass

    print("\nDownloading spaCy English model (en_core_web_sm)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ spaCy en_core_web_sm ready")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ spaCy en_core_web_sm failed: {e.stderr}")
        return False
    except Exception as e:  # noqa: BLE001 - report whatever went wrong
        print(f"✗ spaCy en_core_web_sm failed: {e}")
        return False


def download_nltk_resource(name: str, find_path: str, description: str) -> bool:
    """Download one NLTK resource and confirm it is actually loadable."""
    import nltk  # caller has already checked this imports

    print(f"\nDownloading NLTK {description} ({name})...")
    try:
        nltk.download(name, quiet=True)
        # Some corpora (wordnet) stay zipped and are read lazily, so a bare
        # find('corpora/wordnet') raises even though the corpus works fine.
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.data.find(f"{find_path}.zip")
        print(f"✓ NLTK {name} ready")
        return True
    except LookupError:
        print(f"✗ NLTK {name} downloaded but not found at {find_path}")
        return False
    except Exception as e:  # noqa: BLE001 - report whatever went wrong
        print(f"✗ NLTK {name} failed: {e}")
        return False


def installed(module: str) -> bool:
    """True if a module can be imported without actually importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def main() -> None:
    print("Setting up NLP models and data...")
    print("=" * 60)

    # spaCy and NLTK only ship with phase-2 and later. Running this after
    # `uv sync --group phase-1` is normal and should not be an error.
    if not installed("spacy") and not installed("nltk"):
        print("\nspaCy and NLTK are not installed — nothing to download.")
        print("They arrive with phase 2 onwards:")
        print("  uv sync --group phase-2   # or --all-groups")
        print("\nNothing to do for phase 1. You're set.")
        return

    results = []
    if installed("spacy"):
        results.append(download_spacy_model())
    else:
        print("\n- spaCy not installed, skipping (needed from phase 2 onwards)")

    if installed("nltk"):
        for name, find_path, description in NLTK_RESOURCES:
            results.append(download_nltk_resource(name, find_path, description))
    else:
        print("- NLTK not installed, skipping (needed from phase 2 onwards)")

    success_count = sum(results)
    total_count = len(results)

    print("\n" + "=" * 60)
    print(f"Setup complete: {success_count}/{total_count} items ready")

    if success_count < total_count:
        print("\nSome downloads failed. To retry manually:")
        print("  python -m spacy download en_core_web_sm")
        names = ", ".join(repr(n) for n, _, _ in NLTK_RESOURCES)
        print(f"  python -c \"import nltk; [nltk.download(n) for n in [{names}]]\"")
        sys.exit(1)


if __name__ == "__main__":
    main()

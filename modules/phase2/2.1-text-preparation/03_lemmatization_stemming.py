"""
03 - Lemmatization & Stemming
=============================
Reduce words to their base forms for better matching and analysis.

Key concept: Lemmatization is linguistically accurate; stemming is faster but cruder.

Book reference: NLP_cook.1, speach_lang.I.2.6
"""

import nltk
import spacy

# Download NLTK data
nltk.download("wordnet", quiet=True)
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")


def stem_porter(words: list[str]) -> list[str]:
    """Apply Porter stemmer - oldest and most aggressive."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def stem_snowball(words: list[str]) -> list[str]:
    """Apply Snowball stemmer - improved Porter, less aggressive."""
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]


def lemmatize_nltk(words: list[str]) -> list[str]:
    """Lemmatize using NLTK WordNet - dictionary-based."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def lemmatize_spacy(text: str) -> list[tuple[str, str]]:
    """Lemmatize using spaCy - context-aware, best quality."""
    doc = nlp(text)
    return [(token.text, token.lemma_) for token in doc]


if __name__ == "__main__":
    # Test words showing the differences
    test_words = ["running", "runs", "ran", "better", "studies", "studying",
                  "companies", "programming", "worked", "working"]
    
    print("=== COMPARISON: STEMMING vs LEMMATIZATION ===")
    print(f"{'Word':<15} {'Porter':<15} {'Snowball':<15} {'NLTK Lemma':<15}")
    print("-" * 60)
    
    porter_results = stem_porter(test_words)
    snowball_results = stem_snowball(test_words)
    nltk_lemmas = lemmatize_nltk(test_words)
    
    for word, porter, snowball, lemma in zip(test_words, porter_results, 
                                              snowball_results, nltk_lemmas):
        print(f"{word:<15} {porter:<15} {snowball:<15} {lemma:<15}")
    
    # spaCy lemmatization with context
    print("\n=== SPACY LEMMATIZATION (context-aware) ===")
    text = "The companies are studying better programming practices."
    results = lemmatize_spacy(text)
    for token, lemma in results:
        if token.lower() != lemma:
            print(f"  {token} -> {lemma}")
    
    print("\n=== KEY DIFFERENCES ===")
    print("• Stemming: Fast, but 'studies' -> 'studi' (not a word)")
    print("• Lemmatization: Slower, but 'studies' -> 'study' (correct)")

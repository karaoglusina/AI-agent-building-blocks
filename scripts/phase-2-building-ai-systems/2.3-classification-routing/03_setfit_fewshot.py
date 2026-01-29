"""
03 - SetFit Few-Shot Training
=============================
Train a classifier with minimal examples using SetFit.

Key concept: SetFit achieves good accuracy with just 8-16 examples per class.

Book reference: hands_on_LLM.III.11
"""

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path

# Training data: just 4 examples per class
TRAIN_DATA = {
    "text": [
        # Engineering
        "Senior Python Developer for backend systems",
        "Software Engineer - Full Stack",
        "DevOps Engineer - Kubernetes",
        "Mobile Developer - iOS",
        # Data
        "Data Scientist - Machine Learning",
        "Data Analyst - Business Intelligence",
        "ML Engineer - NLP",
        "Data Engineer - ETL Pipelines",
        # Business
        "Product Manager - Growth",
        "Business Analyst",
        "Project Manager - Agile",
        "Strategy Consultant"],
    "label": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # 0=Engineering, 1=Data, 2=Business
}

LABEL_NAMES = {0: "Engineering", 1: "Data", 2: "Business"}


def train_classifier():
    """Train a SetFit classifier with few examples."""
    print("Training SetFit classifier with 4 examples per class...")
    
    # Create dataset
    train_dataset = Dataset.from_dict(TRAIN_DATA)
    
    # Load pre-trained model
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Training arguments (minimal for demo)
    args = TrainingArguments(
        batch_size=4,
        num_epochs=1)
    
    # Train
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    
    print("Training complete!\n")
    return model


def classify_jobs(model, jobs: list[dict]) -> list[tuple[str, str]]:
    """Classify jobs using trained model."""
    texts = [f"{job['title']} - {job['description'][:200]}" for job in jobs]
    predictions = model.predict(texts)
    return [(job["title"], LABEL_NAMES[pred]) for job, pred in zip(jobs, predictions)]


if __name__ == "__main__":
    print("=== SETFIT FEW-SHOT CLASSIFICATION ===\n")
    
    # Train model
    model = train_classifier()
    
    # Classify real jobs
    jobs = load_sample_jobs(10)
    results = classify_jobs(model, jobs)
    
    print("=== CLASSIFICATION RESULTS ===")
    for title, category in results:
        print(f"  {category:12} | {title}")
    
    print(f"\n✓ Trained with only {len(TRAIN_DATA['text'])} examples total")

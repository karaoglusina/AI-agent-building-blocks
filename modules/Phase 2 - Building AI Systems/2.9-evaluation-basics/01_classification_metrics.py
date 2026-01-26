"""
01 - Classification Metrics
===========================
Precision, recall, F1, and confusion matrix.

Key concept: Different metrics matter for different use cases - pick the right ones.

Book reference: AI_eng.3, speach_lang.I.4.7
"""

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np


def calculate_metrics(y_true: list, y_pred: list, average: str = "binary"):
    """Calculate core classification metrics."""
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def print_confusion_matrix(y_true: list, y_pred: list, labels: list = None):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    print("Confusion Matrix:")
    print("              Predicted")
    print("Actual  ", end="")
    for label in labels:
        print(f"{str(label):>8}", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"{str(label):>8}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>8}", end="")
        print()


# Example: Job classification
# True labels and predictions for job category classification
y_true = [
    "Engineering", "Engineering", "Engineering", "Engineering",
    "Data", "Data", "Data",
    "Product", "Product", "Product",
    "Marketing", "Marketing",
]

y_pred = [
    "Engineering", "Engineering", "Data", "Engineering",  # 1 wrong
    "Data", "Data", "Engineering",  # 1 wrong
    "Product", "Marketing", "Product",  # 1 wrong
    "Marketing", "Sales",  # 1 wrong
]


if __name__ == "__main__":
    print("=== CLASSIFICATION METRICS ===\n")
    
    # Binary example first
    print("--- Binary Classification ---")
    binary_true = [1, 1, 1, 0, 0, 0, 1, 0]
    binary_pred = [1, 1, 0, 0, 0, 1, 1, 0]
    
    metrics = calculate_metrics(binary_true, binary_pred)
    print(f"Precision: {metrics['precision']:.2f} (of predicted positives, how many correct?)")
    print(f"Recall: {metrics['recall']:.2f} (of actual positives, how many found?)")
    print(f"F1 Score: {metrics['f1']:.2f} (harmonic mean of precision and recall)")
    
    print("\n--- Multi-class Classification ---")
    print("(Job category prediction)\n")
    
    # Confusion matrix
    labels = ["Engineering", "Data", "Product", "Marketing", "Sales"]
    print_confusion_matrix(y_true, y_pred, labels)
    
    # Full report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # When to use which metric
    print("=== METRIC SELECTION GUIDE ===")
    print("• High precision needed: Spam filter (don't block good emails)")
    print("• High recall needed: Disease detection (don't miss cases)")
    print("• F1 score: Balanced importance of both")
    print("• Accuracy: Only if classes are balanced")

"""
03 - Context Prioritization
===========================
Decide what to include when space is limited.

Key concept: Not all context is equal - prioritize by relevance and recency.

Book reference: AI_eng.5, AI_eng.6
"""

import tiktoken
from dataclasses import dataclass

encoding = tiktoken.encoding_for_model("gpt-4o-mini")


@dataclass
class ContextItem:
    """An item that could be included in context."""
    content: str
    priority: int  # 1 = highest, 5 = lowest
    category: str  # "system", "history", "retrieval", "user"
    
    @property
    def tokens(self) -> int:
        return len(encoding.encode(self.content))


def prioritize_context(
    items: list[ContextItem],
    max_tokens: int,
    reserve_for_response: int = 500
) -> list[ContextItem]:
    """Select items that fit within token budget, prioritizing important items."""
    available = max_tokens - reserve_for_response
    
    # Sort by priority (lower = more important), then by category importance
    category_order = {"system": 0, "user": 1, "retrieval": 2, "history": 3}
    sorted_items = sorted(items, key=lambda x: (x.priority, category_order.get(x.category, 4)))
    
    selected = []
    used_tokens = 0
    
    for item in sorted_items:
        if used_tokens + item.tokens <= available:
            selected.append(item)
            used_tokens += item.tokens
        else:
            print(f"  Dropped ({item.tokens} tokens): {item.content[:50]}...")
    
    return selected


def build_context(items: list[ContextItem]) -> str:
    """Build context string from selected items."""
    # Group by category
    by_category = {}
    for item in items:
        by_category.setdefault(item.category, []).append(item)
    
    parts = []
    for category in ["system", "retrieval", "history", "user"]:
        if category in by_category:
            for item in by_category[category]:
                parts.append(item.content)
    
    return "\n\n".join(parts)


if __name__ == "__main__":
    print("=== CONTEXT PRIORITIZATION ===\n")
    
    # Sample context items
    items = [
        ContextItem("You are a job search assistant.", priority=1, category="system"),
        ContextItem("Be concise and helpful.", priority=1, category="system"),
        ContextItem("User previously asked about Python jobs.", priority=3, category="history"),
        ContextItem("User mentioned preferring remote work.", priority=2, category="history"),
        ContextItem("Retrieved: Senior Python Developer at TechCorp - " + "x" * 200, priority=2, category="retrieval"),
        ContextItem("Retrieved: Python Engineer at StartupXYZ - " + "x" * 200, priority=2, category="retrieval"),
        ContextItem("Retrieved: Data Scientist at DataCo - " + "x" * 200, priority=3, category="retrieval"),
        ContextItem("Find me Python jobs in Amsterdam", priority=1, category="user"),
    ]
    
    # Show all items
    total_tokens = sum(item.tokens for item in items)
    print(f"Total items: {len(items)}, Total tokens: {total_tokens}")
    print()
    
    # Prioritize with limited budget
    max_budget = 200
    print(f"Budget: {max_budget} tokens (reserving 50 for response)\n")
    
    selected = prioritize_context(items, max_tokens=max_budget, reserve_for_response=50)
    
    print(f"\nSelected {len(selected)} items:")
    for item in selected:
        print(f"  [{item.priority}] {item.category}: {item.content[:60]}... ({item.tokens} tokens)")
    
    print(f"\n=== BUILT CONTEXT ===")
    context = build_context(selected)
    print(context)
    print(f"\nTotal context tokens: {len(encoding.encode(context))}")

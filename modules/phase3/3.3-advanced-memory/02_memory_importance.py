"""
02 - Memory Importance Scoring
===============================
Prioritize important memories.

Key concept: Score memories by importance to decide what to keep and what to discard.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from openai import OpenAI
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Literal

client = OpenAI()


class Memory(BaseModel):
    """A memory with importance metadata."""
    content: str
    category: Literal["fact", "preference", "goal", "event"]
    importance_score: float = Field(ge=0.0, le=1.0, description="0=trivial, 1=critical")
    created_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.now)


class MemoryWithScore(BaseModel):
    """Memory extraction with importance scoring."""
    memories: list[Memory]


class ImportanceBasedMemory:
    """Memory system that prioritizes by importance."""

    def __init__(self, max_memories: int = 10):
        self.max_memories = max_memories
        self.memories: list[Memory] = []

    def extract_memories(self, text: str) -> list[Memory]:
        """Extract memories with importance scores."""
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": "Extract important information from the text. "
                               "Score importance: 0.9-1.0 = critical long-term info, "
                               "0.6-0.8 = useful preferences/facts, "
                               "0.3-0.5 = contextual info, "
                               "0.0-0.2 = trivial/temporary. "
                               "Categories: fact (biographical), preference (likes/dislikes), "
                               "goal (objectives), event (something that happened)."
                },
                {"role": "user", "content": text}
            ],
            text_format=MemoryWithScore
        )
        return response.output_parsed.memories

    def add_memories(self, new_memories: list[Memory]):
        """Add memories and prune if over limit."""
        self.memories.extend(new_memories)
        self._prune_memories()

    def _calculate_retention_score(self, memory: Memory) -> float:
        """Calculate composite score for retention priority."""
        # Start with importance
        score = memory.importance_score

        # Boost by access frequency (max +0.2)
        access_boost = min(0.2, memory.access_count * 0.05)
        score += access_boost

        # Decay by age (reduce by up to 0.15 over 30 days)
        age_days = (datetime.now() - memory.created_at).days
        age_penalty = min(0.15, age_days / 30 * 0.15)
        score -= age_penalty

        # Slight boost for recent access (max +0.1)
        days_since_access = (datetime.now() - memory.last_accessed).days
        recency_boost = max(0, 0.1 - (days_since_access * 0.01))
        score += recency_boost

        return max(0, min(1, score))  # Clamp to [0, 1]

    def _prune_memories(self):
        """Keep only top memories by retention score."""
        if len(self.memories) <= self.max_memories:
            return

        # Calculate retention scores
        scored = [(mem, self._calculate_retention_score(mem)) for mem in self.memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top N
        self.memories = [mem for mem, score in scored[:self.max_memories]]

    def access_memory(self, memory: Memory):
        """Mark a memory as accessed (boosts retention)."""
        memory.access_count += 1
        memory.last_accessed = datetime.now()

    def get_memories_by_importance(self, min_score: float = 0.0) -> list[Memory]:
        """Get memories above importance threshold."""
        return [m for m in self.memories if m.importance_score >= min_score]

    def get_summary(self) -> str:
        """Get summary with retention scores."""
        if not self.memories:
            return "No memories stored."

        lines = []
        for mem in sorted(self.memories, key=self._calculate_retention_score, reverse=True):
            retention = self._calculate_retention_score(mem)
            age = (datetime.now() - mem.created_at).days
            lines.append(
                f"[{mem.category}] {mem.content[:60]}... "
                f"(imp={mem.importance_score:.2f}, ret={retention:.2f}, age={age}d)"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== MEMORY IMPORTANCE SCORING ===\n")

    memory = ImportanceBasedMemory(max_memories=5)

    # Process multiple conversations
    conversations = [
        "I'm looking for a Python job in Amsterdam.",
        "I graduated from MIT with a CS degree in 2018.",
        "It's a nice day today.",
        "My ultimate career goal is to become a CTO.",
        "I had coffee this morning.",
        "I have 6 years of experience building distributed systems.",
        "The weather is cloudy.",
    ]

    print("Processing conversations and extracting memories...")
    for i, conv in enumerate(conversations, 1):
        print(f"  [{i}] {conv}")
        extracted = memory.extract_memories(conv)
        memory.add_memories(extracted)

    print(f"\n=== RETAINED MEMORIES (top {memory.max_memories}) ===\n")
    print(memory.get_summary())

    print("\n=== HIGH-IMPORTANCE ONLY (>= 0.7) ===\n")
    high_importance = memory.get_memories_by_importance(0.7)
    for mem in high_importance:
        print(f"  • {mem.content} (score: {mem.importance_score:.2f})")

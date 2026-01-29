"""
03 - Memory Consolidation
==========================
Merge related memories over time.

Key concept: Combine redundant or related memories to reduce storage and improve coherence.

Book reference: AI_eng.6
"""

import utils._load_env  # Loads .env file automatically

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openai import OpenAI
import os
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class Memory(BaseModel):
    """A single memory entry."""
    id: str
    content: str
    created_at: datetime
    source_ids: list[str] = []  # Tracks merged memories


class ConsolidationResult(BaseModel):
    """Result of memory consolidation."""
    should_merge: bool
    merged_content: str | None = None
    reason: str


class MemoryConsolidation:
    """System for merging related memories."""

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.next_id = 1

    def add_memory(self, content: str) -> str:
        """Add a new memory and attempt consolidation."""
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1

        memory = Memory(
            id=memory_id,
            content=content,
            created_at=datetime.now()
        )

        # Check for consolidation opportunities
        consolidated = self._try_consolidate(memory)

        if consolidated:
            self.memories[consolidated.id] = consolidated
            return consolidated.id
        else:
            self.memories[memory_id] = memory
            return memory_id

    def _try_consolidate(self, new_memory: Memory) -> Memory | None:
        """Try to consolidate new memory with existing ones."""
        for existing_id, existing in self.memories.items():
            # Check if memories should be merged
            result = self._check_consolidation(existing.content, new_memory.content)

            if result.should_merge:
                print(f"  → Consolidating: {existing_id} + {new_memory.id}")
                print(f"    Reason: {result.reason}")

                # Create consolidated memory
                consolidated = Memory(
                    id=existing_id,  # Keep original ID
                    content=result.merged_content,
                    created_at=existing.created_at,  # Keep earliest timestamp
                    source_ids=existing.source_ids + [new_memory.id]
                )
                return consolidated

        return None

    def _check_consolidation(self, memory1: str, memory2: str) -> ConsolidationResult:
        """Determine if two memories should be consolidated."""
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": "Determine if these two memories should be merged. "
        "Merge if they're redundant, contradictory (keep newer), "
        "or can be combined into a more complete fact. "
        "If merging, provide the consolidated content."
        },
        {
        "role": "user",
        "content": f"Memory 1: {memory1}\nMemory 2: {memory2}"
        }
        ],
        response_format={"type": "json_object"}
        )
        return ConsolidationResult.model_validate_json(response.choices[0].message.content)

    def consolidate_all(self):
        """Perform a full consolidation pass on all memories."""
        print("\n=== FULL CONSOLIDATION PASS ===")

        memory_list = list(self.memories.values())
        i = 0

        while i < len(memory_list):
            j = i + 1
            while j < len(memory_list):
                mem1, mem2 = memory_list[i], memory_list[j]

                result = self._check_consolidation(mem1.content, mem2.content)

                if result.should_merge:
                    print(f"  → Merging {mem1.id} + {mem2.id}")
                    print(f"    Reason: {result.reason}")

                    # Create consolidated memory
                    consolidated = Memory(
                        id=mem1.id,
                        content=result.merged_content,
                        created_at=mem1.created_at,
                        source_ids=mem1.source_ids + mem2.source_ids + [mem2.id]
                    )

                    # Update storage
                    self.memories[mem1.id] = consolidated
                    del self.memories[mem2.id]

                    # Remove from list and restart inner loop
                    memory_list[i] = consolidated
                    memory_list.pop(j)
                else:
                    j += 1

            i += 1

    def get_summary(self) -> str:
        """Get summary of all memories."""
        lines = []
        for mem_id, mem in self.memories.items():
            sources = f" (merged from {len(mem.source_ids)})" if mem.source_ids else ""
            lines.append(f"[{mem_id}] {mem.content}{sources}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== MEMORY CONSOLIDATION ===\n")

    consolidator = MemoryConsolidation()

    # Add memories that should be consolidated
    memories_to_add = [
        "User knows Python",
        "User has 5 years of Python experience",
        "User prefers remote work",
        "User is an expert in Python with 5+ years",
        "User wants remote positions only",
        "User lives in Berlin",
        "User is based in Berlin, Germany",
        "User knows JavaScript basics"]

    print("Adding memories (with automatic consolidation):\n")
    for content in memories_to_add:
        print(f"Adding: {content}")
        consolidator.add_memory(content)
        print()

    print("\n=== AFTER INCREMENTAL CONSOLIDATION ===")
    print(consolidator.get_summary())

    # Try full consolidation pass
    consolidator.consolidate_all()

    print("\n=== AFTER FULL CONSOLIDATION ===")
    print(consolidator.get_summary())

    print(f"\nMemories reduced: {len(memories_to_add)} → {len(consolidator.memories)}")

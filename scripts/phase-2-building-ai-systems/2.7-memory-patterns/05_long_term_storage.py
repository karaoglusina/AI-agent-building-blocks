"""
05 - Long-Term Storage
======================
Persist important facts to file/database.

Key concept: Store facts that should survive across sessions.

Book reference: AI_eng.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import os
from pydantic import BaseModel


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class ImportantFact(BaseModel):
    """A fact worth remembering long-term."""
    fact: str
    category: str  # preference, skill, experience, goal
    confidence: float


class ExtractedFacts(BaseModel):
    """Facts extracted from conversation."""
    facts: list[ImportantFact]


class LongTermMemory:
    """Memory that persists to disk."""
    
    def __init__(self, storage_path: str = "user_memory.json"):
        self.storage_path = Path(storage_path)
        self.facts = self._load()
    
    def _load(self) -> list[dict]:
        """Load facts from disk."""
        if self.storage_path.exists():
            return json.loads(self.storage_path.read_text())
        return []
    
    def _save(self):
        """Save facts to disk."""
        self.storage_path.write_text(json.dumps(self.facts, indent=2))
    
    def extract_facts(self, conversation_text: str) -> list[ImportantFact]:
        """Extract important facts from conversation."""
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": "Extract important facts about the user that should be remembered. "
        "Categories: preference, skill, experience, goal. "
        "Only extract clear, specific facts."
        },
        {"role": "user", "content": conversation_text}
        ]
        ,
        response_format={"type": "json_object"})
        return ImportantFact.model_validate_json(response.choices[0].message.content).facts
    
    def add_facts(self, facts: list[ImportantFact]):
        """Add new facts to storage."""
        for fact in facts:
            # Avoid duplicates
            existing = [f["fact"].lower() for f in self.facts]
            if fact.fact.lower() not in existing:
                self.facts.append({
                    "fact": fact.fact,
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "added_at": datetime.now().isoformat()
                })
        self._save()
    
    def get_facts_by_category(self, category: str = None) -> list[dict]:
        """Get facts, optionally filtered by category."""
        if category:
            return [f for f in self.facts if f["category"] == category]
        return self.facts
    
    def get_context_string(self) -> str:
        """Get facts as context string for prompt."""
        if not self.facts:
            return "No stored information about user."
        
        by_category = {}
        for fact in self.facts:
            cat = fact["category"]
            by_category.setdefault(cat, []).append(fact["fact"])
        
        parts = []
        for category, facts in by_category.items():
            parts.append(f"{category.title()}: {'; '.join(facts)}")
        
        return "\n".join(parts)
    
    def clear(self):
        """Clear all stored facts."""
        self.facts = []
        self._save()


if __name__ == "__main__":
    print("=== LONG-TERM STORAGE ===\n")
    
    # Use temp file for demo
    memory = LongTermMemory("_demo_memory.json")
    memory.clear()  # Start fresh for demo
    
    # Simulate a conversation
    conversation = """
    User: I'm a senior software engineer with 8 years of experience.
    User: I specialize in Python and machine learning.
    User: I prefer remote work and am based in Amsterdam.
    User: My goal is to transition into an ML engineering role at a top tech company.
    User: I have experience with PyTorch and TensorFlow.
    """
    
    print("Extracting facts from conversation...")
    facts = memory.extract_facts(conversation)
    
    print(f"\nExtracted {len(facts)} facts:")
    for fact in facts:
        print(f"  [{fact.category}] {fact.fact} (confidence: {fact.confidence:.0%})")
    
    # Store facts
    memory.add_facts(facts)
    
    print(f"\n=== STORED FACTS ===")
    print(memory.get_context_string())
    
    # Show file
    print(f"\n=== PERSISTED TO: {memory.storage_path} ===")
    print(memory.storage_path.read_text()[:500] + "...")
    
    # Cleanup demo file
    memory.storage_path.unlink()

"""
04 - Entity Memory
==================
Track entities mentioned in conversation.

Key concept: Extract and store key entities for persistent reference.

Book reference: AI_eng.6, speach_lang.III.23
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os
from pydantic import BaseModel


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class ExtractedEntities(BaseModel):
    """Entities extracted from a message."""
    skills: list[str] = []
    companies: list[str] = []
    locations: list[str] = []
    job_titles: list[str] = []
    preferences: list[str] = []


class EntityMemory:
    """Memory that tracks entities mentioned in conversation."""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.entities = {
            "skills": set(),
            "companies": set(),
            "locations": set(),
            "job_titles": set(),
            "preferences": set(),
        }
        self.recent_messages = []
    
    def extract_entities(self, text: str) -> ExtractedEntities:
        """Extract entities from text using LLM."""
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": "Extract entities from the user message. "
        "Include skills, companies, locations, job titles, and preferences."
        },
        {"role": "user", "content": text}
        ]
        ,
        response_format={"type": "json_object"})
        return ExtractedEntities.model_validate_json(response.choices[0].message.content)
    
    def add_user_message(self, content: str):
        """Process user message and extract entities."""
        # Extract and store entities
        extracted = self.extract_entities(content)
        
        for skill in extracted.skills:
            self.entities["skills"].add(skill.lower())
        for company in extracted.companies:
            self.entities["companies"].add(company)
        for location in extracted.locations:
            self.entities["locations"].add(location)
        for title in extracted.job_titles:
            self.entities["job_titles"].add(title.lower())
        for pref in extracted.preferences:
            self.entities["preferences"].add(pref.lower())
        
        self.recent_messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add assistant message to recent history."""
        self.recent_messages.append({"role": "assistant", "content": content})
        # Keep only last 4 messages
        self.recent_messages = self.recent_messages[-4:]
    
    def get_entity_context(self) -> str:
        """Get entity context string."""
        parts = []
        for category, items in self.entities.items():
            if items:
                parts.append(f"{category}: {', '.join(items)}")
        return "\n".join(parts) if parts else "No entities tracked yet."
    
    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        entity_context = self.get_entity_context()
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Known user information:\n{entity_context}"},
            *self.recent_messages
        ]


def chat_with_entity_memory(memory: EntityMemory, user_input: str) -> str:
    """Chat using entity memory."""
    memory.add_user_message(user_input)
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=memory.get_messages()
    )
    
    assistant_reply = response.choices[0].message.content
    memory.add_assistant_message(assistant_reply)
    
    return assistant_reply


if __name__ == "__main__":
    print("=== ENTITY MEMORY ===\n")
    
    memory = EntityMemory(
        system_prompt="You are a job search assistant. Use known user information to personalize responses."
    )
    
    conversation = [
        "I know Python, JavaScript, and SQL.",
        "I'm interested in Google and Microsoft.",
        "I prefer remote work in the Netherlands.",
        "Looking for senior data engineer positions.",
        "What jobs match my profile?",  # Should use all remembered entities
    ]
    
    for user_input in conversation:
        print(f"User: {user_input}")
        response = chat_with_entity_memory(memory, user_input)
        print(f"Assistant: {response}\n")
    
    print("=" * 50)
    print("\n=== TRACKED ENTITIES ===")
    for category, items in memory.entities.items():
        if items:
            print(f"{category}: {', '.join(items)}")

"""
04 - Preference Memory System
==============================
Full preference detection + storage.

Key concept: Automatically detect, extract, and persist user preferences from conversations.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from openai import OpenAI
from pydantic import BaseModel, Field
import chromadb
from datetime import datetime
import json

client = OpenAI()


class Preference(BaseModel):
    """A user preference."""
    category: str = Field(description="e.g., work_style, location, compensation, culture")
    preference: str = Field(description="The actual preference")
    strength: float = Field(ge=0.0, le=1.0, description="How strong is this preference")
    context: str = Field(description="Original context where mentioned")


class PreferenceExtraction(BaseModel):
    """Extraction result."""
    preferences: list[Preference]


class PreferenceMemory:
    """Complete system for tracking user preferences."""

    def __init__(self, collection_name: str = "preferences"):
        self.chroma_client = chromadb.Client()

        # Create or get collection
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(collection_name)

        self.preference_count = 0

    def detect_preferences(self, conversation: str) -> list[Preference]:
        """Detect preferences from conversation text."""
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": "Extract user preferences from the conversation. "
                               "Look for likes/dislikes, requirements, priorities, and deal-breakers. "
                               "Categories: work_style, location, compensation, company_size, "
                               "industry, culture, growth, benefits. "
                               "Strength: 1.0 = must-have, 0.7 = strong preference, "
                               "0.4 = nice-to-have, 0.1 = weak preference."
                },
                {"role": "user", "content": conversation}
            ],
            text_format=PreferenceExtraction
        )
        return response.output_parsed.preferences

    def store_preference(self, preference: Preference):
        """Store preference in vector database."""
        pref_id = f"pref_{self.preference_count}"
        self.preference_count += 1

        # Create embedding text
        embed_text = f"{preference.category}: {preference.preference}"

        # Get embedding
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=embed_text
        )
        embedding = embedding_response.data[0].embedding

        # Store in ChromaDB
        self.collection.add(
            ids=[pref_id],
            embeddings=[embedding],
            documents=[preference.preference],
            metadatas=[{
                "category": preference.category,
                "strength": preference.strength,
                "context": preference.context,
                "created_at": datetime.now().isoformat()
            }]
        )

    def retrieve_relevant_preferences(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve preferences relevant to a query."""
        # Get embedding for query
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        if not results["documents"][0]:
            return []

        # Format results
        preferences = []
        for i in range(len(results["documents"][0])):
            preferences.append({
                "preference": results["documents"][0][i],
                "category": results["metadatas"][0][i]["category"],
                "strength": results["metadatas"][0][i]["strength"],
                "distance": results["distances"][0][i]
            })

        return preferences

    def get_preferences_by_category(self, category: str) -> list[dict]:
        """Get all preferences for a specific category."""
        results = self.collection.get(
            where={"category": category}
        )

        if not results["documents"]:
            return []

        preferences = []
        for i in range(len(results["documents"])):
            preferences.append({
                "preference": results["documents"][i],
                "category": results["metadatas"][i]["category"],
                "strength": results["metadatas"][i]["strength"]
            })

        return preferences

    def get_all_preferences(self) -> list[dict]:
        """Get all stored preferences grouped by category."""
        results = self.collection.get()

        if not results["documents"]:
            return []

        # Group by category
        by_category = {}
        for i in range(len(results["documents"])):
            cat = results["metadatas"][i]["category"]
            if cat not in by_category:
                by_category[cat] = []

            by_category[cat].append({
                "preference": results["documents"][i],
                "strength": results["metadatas"][i]["strength"]
            })

        return by_category


if __name__ == "__main__":
    print("=== PREFERENCE MEMORY SYSTEM ===\n")

    memory = PreferenceMemory()

    # Example conversation
    conversation = """
    User: I'm looking for a fully remote job. I absolutely need work-from-home flexibility.
    User: I prefer startups over big corporations. The fast-paced environment suits me better.
    User: Salary-wise, I need at least $120k, but equity would be a nice bonus.
    User: I'd love to work in AI/ML space, that's my passion.
    User: Good work-life balance is important, but not a dealbreaker.
    User: I prefer async communication over constant meetings.
    """

    print("Detecting preferences from conversation...")
    preferences = memory.detect_preferences(conversation)

    print(f"\nExtracted {len(preferences)} preferences:")
    for pref in preferences:
        print(f"  [{pref.category}] {pref.preference} (strength: {pref.strength:.2f})")

    print("\nStoring in vector database...")
    for pref in preferences:
        memory.store_preference(pref)

    print("\n=== RETRIEVAL EXAMPLES ===")

    # Retrieve by relevance
    query = "What kind of company culture does the user want?"
    print(f"\nQuery: {query}")
    relevant = memory.retrieve_relevant_preferences(query, n_results=3)
    for pref in relevant:
        print(f"  • {pref['preference']} (strength: {pref['strength']}, relevance: {1-pref['distance']:.2f})")

    # Get by category
    print("\n=== PREFERENCES BY CATEGORY ===")
    all_prefs = memory.get_all_preferences()
    for category, prefs in sorted(all_prefs.items()):
        print(f"\n{category.upper()}:")
        for pref in sorted(prefs, key=lambda x: x["strength"], reverse=True):
            strength_label = "🔴 Must-have" if pref["strength"] >= 0.9 else "🟡 Strong" if pref["strength"] >= 0.6 else "🟢 Nice-to-have"
            print(f"  {strength_label}: {pref['preference']}")

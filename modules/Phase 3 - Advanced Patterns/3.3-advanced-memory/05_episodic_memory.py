"""
05 - Episodic Memory
====================
Remember specific past interactions.

Key concept: Store and retrieve specific episodes (conversations/events) with context and timeline.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from openai import OpenAI
from pydantic import BaseModel
import chromadb
from datetime import datetime
from typing import Literal

client = OpenAI()


class Episode(BaseModel):
    """A specific interaction or event."""
    title: str
    summary: str
    event_type: Literal["conversation", "action", "outcome", "feedback"]
    participants: list[str] = []
    key_points: list[str] = []
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    timestamp: datetime


class EpisodeExtraction(BaseModel):
    """Extraction from conversation."""
    episode: Episode


class EpisodicMemory:
    """System for storing and recalling specific past interactions."""

    def __init__(self, collection_name: str = "episodes"):
        self.chroma_client = chromadb.Client()

        # Create collection
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(collection_name)

        self.episode_count = 0

    def create_episode(self, conversation: str, context: str = "") -> Episode:
        """Create an episode from a conversation."""
        prompt = f"Create an episode summary from this conversation.\n\n{conversation}"
        if context:
            prompt += f"\n\nContext: {context}"

        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": "Extract a structured episode from the conversation. "
                               "Title should be concise. Summary should capture the key interaction. "
                               "Identify participants (user, assistant, other names mentioned). "
                               "Extract 3-5 key points. Determine overall sentiment."
                },
                {"role": "user", "content": prompt}
            ],
            text_format=EpisodeExtraction
        )

        episode = response.output_parsed.episode
        episode.timestamp = datetime.now()
        return episode

    def store_episode(self, episode: Episode) -> str:
        """Store episode in memory."""
        episode_id = f"ep_{self.episode_count}"
        self.episode_count += 1

        # Create rich text for embedding
        embed_text = (
            f"{episode.title}. {episode.summary}. "
            f"Key points: {', '.join(episode.key_points)}"
        )

        # Get embedding
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=embed_text
        )
        embedding = embedding_response.data[0].embedding

        # Store in ChromaDB
        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            documents=[episode.summary],
            metadatas=[{
                "title": episode.title,
                "event_type": episode.event_type,
                "participants": ",".join(episode.participants),
                "key_points": "|".join(episode.key_points),
                "sentiment": episode.sentiment,
                "timestamp": episode.timestamp.isoformat()
            }]
        )

        return episode_id

    def recall_similar_episodes(self, query: str, n_results: int = 3) -> list[dict]:
        """Recall episodes similar to a query."""
        # Get query embedding
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
        episodes = []
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            episodes.append({
                "title": meta["title"],
                "summary": results["documents"][0][i],
                "event_type": meta["event_type"],
                "key_points": meta["key_points"].split("|"),
                "sentiment": meta["sentiment"],
                "timestamp": meta["timestamp"],
                "relevance": 1 - results["distances"][0][i]
            })

        return episodes

    def get_recent_episodes(self, n: int = 5) -> list[dict]:
        """Get the N most recent episodes."""
        results = self.collection.get()

        if not results["documents"]:
            return []

        # Parse and sort by timestamp
        episodes = []
        for i in range(len(results["documents"])):
            meta = results["metadatas"][i]
            episodes.append({
                "title": meta["title"],
                "summary": results["documents"][i],
                "event_type": meta["event_type"],
                "timestamp": datetime.fromisoformat(meta["timestamp"])
            })

        episodes.sort(key=lambda x: x["timestamp"], reverse=True)
        return episodes[:n]

    def get_episodes_by_type(self, event_type: str) -> list[dict]:
        """Get episodes of a specific type."""
        results = self.collection.get(
            where={"event_type": event_type}
        )

        if not results["documents"]:
            return []

        episodes = []
        for i in range(len(results["documents"])):
            meta = results["metadatas"][i]
            episodes.append({
                "title": meta["title"],
                "summary": results["documents"][i],
                "timestamp": meta["timestamp"]
            })

        return episodes


if __name__ == "__main__":
    print("=== EPISODIC MEMORY ===\n")

    memory = EpisodicMemory()

    # Simulate different episodes
    episodes_data = [
        {
            "conversation": """
            User: I just had an interview at Google for a senior engineer role.
            Assistant: How did it go?
            User: Really well! I aced the system design round. They asked about scaling
            a distributed cache and I discussed my experience with Redis.
            """,
            "context": "Job interview follow-up"
        },
        {
            "conversation": """
            User: I'm frustrated. I applied to 20 companies this week and only got 2 responses.
            Assistant: That's tough, but it's a normal response rate. Let's refine your approach.
            User: You're right. Maybe I need to tailor my applications more.
            """,
            "context": "Application strategy discussion"
        },
        {
            "conversation": """
            User: Good news! I got an offer from the startup I interviewed with.
            User: They're offering $140k base + equity. I'm excited!
            Assistant: Congratulations! That's a great offer.
            """,
            "context": "Job offer received"
        },
    ]

    print("Creating and storing episodes...\n")
    for i, ep_data in enumerate(episodes_data, 1):
        print(f"Episode {i}:")
        episode = memory.create_episode(ep_data["conversation"], ep_data["context"])
        print(f"  Title: {episode.title}")
        print(f"  Type: {episode.event_type}")
        print(f"  Sentiment: {episode.sentiment}")

        episode_id = memory.store_episode(episode)
        print(f"  Stored as: {episode_id}\n")

    # Recall similar episodes
    print("=== RECALL BY SIMILARITY ===\n")
    query = "Tell me about my interview experiences"
    print(f"Query: {query}\n")

    similar = memory.recall_similar_episodes(query, n_results=2)
    for ep in similar:
        print(f"📅 {ep['title']} (relevance: {ep['relevance']:.2f})")
        print(f"   {ep['summary'][:100]}...")
        print(f"   Sentiment: {ep['sentiment']}\n")

    # Get recent episodes
    print("=== RECENT EPISODES ===\n")
    recent = memory.get_recent_episodes(n=3)
    for ep in recent:
        print(f"📅 [{ep['event_type']}] {ep['title']}")
        print(f"   {ep['timestamp'].strftime('%Y-%m-%d %H:%M')}\n")

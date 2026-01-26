"""
06 - Memory Retrieval
=====================
Retrieve relevant memories for current context.

Key concept: Store memories as embeddings, retrieve based on relevance.

Book reference: AI_eng.6
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from datetime import datetime

client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()


class RetrievalMemory:
    """Memory that retrieves relevant past interactions."""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.collection = chroma.get_or_create_collection(
            name=f"memory_{user_id}",
            embedding_function=openai_ef
        )
        self.message_count = 0
    
    def store_interaction(self, user_message: str, assistant_response: str):
        """Store an interaction for later retrieval."""
        self.message_count += 1
        
        # Store the full interaction
        interaction = f"User: {user_message}\nAssistant: {assistant_response}"
        
        self.collection.add(
            documents=[interaction],
            ids=[f"msg_{self.message_count}"],
            metadatas=[{
                "user_message": user_message[:200],
                "timestamp": datetime.now().isoformat(),
                "type": "interaction"
            }]
        )
    
    def retrieve_relevant(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieve memories relevant to the current query."""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        return results["documents"][0] if results["documents"] else []
    
    def get_memory_context(self, current_query: str) -> str:
        """Get relevant memories as context string."""
        memories = self.retrieve_relevant(current_query)
        
        if not memories:
            return ""
        
        context = "Relevant past interactions:\n"
        for i, mem in enumerate(memories, 1):
            context += f"\n[Memory {i}]\n{mem}\n"
        
        return context


def chat_with_retrieval_memory(memory: RetrievalMemory, user_input: str) -> str:
    """Chat using retrieval-based memory."""
    
    # Get relevant memories
    memory_context = memory.get_memory_context(user_input)
    
    messages = [
        {"role": "system", "content": "You are a job search assistant. Use relevant past interactions to provide personalized help."},
    ]
    
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )
    
    assistant_reply = response.output_text
    
    # Store this interaction
    memory.store_interaction(user_input, assistant_reply)
    
    return assistant_reply


if __name__ == "__main__":
    print("=== MEMORY RETRIEVAL ===\n")
    
    memory = RetrievalMemory(user_id="demo_user")
    
    # First session: build up memories
    session1 = [
        "I'm interested in Python developer positions.",
        "I prefer remote work.",
        "I have 5 years of experience with Django.",
    ]
    
    print("--- Session 1: Building memories ---")
    for msg in session1:
        print(f"User: {msg}")
        response = chat_with_retrieval_memory(memory, msg)
        print(f"Assistant: {response[:100]}...\n")
    
    print(f"Stored {memory.collection.count()} memories\n")
    
    # Second session: memories should be retrieved
    print("--- Session 2: Using memories ---")
    
    queries = [
        "What kind of jobs should I look for?",  # Should retrieve Python/remote preferences
        "Do I have enough experience?",  # Should retrieve 5 years experience
    ]
    
    for query in queries:
        print(f"User: {query}")
        
        # Show what memories are retrieved
        relevant = memory.retrieve_relevant(query)
        if relevant:
            print(f"  [Retrieved {len(relevant)} relevant memories]")
        
        response = chat_with_retrieval_memory(memory, query)
        print(f"Assistant: {response}\n")

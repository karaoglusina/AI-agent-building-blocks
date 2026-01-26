"""
05 - Interactive Chat
=====================
A simple interactive chat loop for testing.

Key concept: Basic REPL pattern for chatbot development.
"""

from openai import OpenAI

client = OpenAI()


def interactive_chat(system_prompt: str):
    """Run an interactive chat session."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    print("=" * 50)
    print("Interactive Chat (type 'quit' to exit)")
    print("=" * 50)
    print()
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        # Check for exit
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        if not user_input:
            continue
        
        # Add to conversation
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
        )
        
        assistant_message = response.output_text
        messages.append({"role": "assistant", "content": assistant_message})
        
        print(f"\nAssistant: {assistant_message}\n")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    interactive_chat(
        system_prompt="""You are a job market analyst specializing in tech roles in the Netherlands.
        
Help users understand:
- Job market trends
- Required skills for different roles
- Salary expectations
- Career advice

Be concise but helpful."""
    )

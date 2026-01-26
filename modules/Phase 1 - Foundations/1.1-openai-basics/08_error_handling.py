"""
08 - Error Handling
===================
Common errors and how to handle them.

Key concept: Always wrap API calls in try/except for production code.
"""

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

client = OpenAI()


def safe_api_call(prompt: str) -> str | None:
    """Make an API call with proper error handling."""
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return response.output_text
    
    except RateLimitError:
        # Too many requests - wait and retry
        print("Rate limited! Wait a moment and retry.")
        return None
    
    except APIConnectionError:
        # Network issues
        print("Connection failed! Check your internet.")
        return None
    
    except APIError as e:
        # Other API errors (invalid request, server error, etc.)
        print(f"API Error: {e}")
        return None


# Test with valid request
result = safe_api_call("Hello!")
if result:
    print(f"Success: {result[:50]}...")


# Common errors you might encounter:
# - AuthenticationError: Invalid API key
# - RateLimitError: Too many requests
# - APIConnectionError: Network issues  
# - BadRequestError: Invalid parameters
# - InternalServerError: OpenAI server issues

"""
08 - Model Gateway Pattern
===========================
Centralized LLM access with controls.

Key concept: A model gateway provides a single point of control for all LLM calls - enabling consistent guardrails, monitoring, rate limiting, and cost tracking across your application.

Book reference: AI_eng.10.3
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
import time

client = OpenAI()


@dataclass
class ModelRequest:
    """Request to the model gateway."""
    user_id: str
    message: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from the model gateway."""
    success: bool
    content: Optional[str]
    error: Optional[str]
    tokens_used: int
    cost: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Simple rate limiter."""

    def __init__(self, max_requests_per_minute: int = 10):
        self.max_requests = max_requests_per_minute
        self.requests = {}  # user_id -> list of timestamps

    def check_rate_limit(self, user_id: str) -> tuple[bool, Optional[str]]:
        """Check if user is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                ts for ts in self.requests[user_id] if ts > minute_ago
            ]
        else:
            self.requests[user_id] = []

        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False, f"Rate limit exceeded: {self.max_requests} requests per minute"

        # Record request
        self.requests[user_id].append(now)
        return True, None


class CostTracker:
    """Track LLM costs per user."""

    def __init__(self):
        self.costs = {}  # user_id -> total_cost
        self.usage = {}  # user_id -> total_tokens

    def record_usage(self, user_id: str, tokens: int, cost: float):
        """Record usage for a user."""
        self.costs[user_id] = self.costs.get(user_id, 0.0) + cost
        self.usage[user_id] = self.usage.get(user_id, 0) + tokens

    def get_user_cost(self, user_id: str) -> float:
        """Get total cost for user."""
        return self.costs.get(user_id, 0.0)

    def get_user_usage(self, user_id: str) -> int:
        """Get total tokens for user."""
        return self.usage.get(user_id, 0)

    def check_budget(self, user_id: str, budget: float) -> bool:
        """Check if user is within budget."""
        return self.get_user_cost(user_id) < budget


class ModelGateway:
    """Centralized gateway for all LLM calls."""

    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests_per_minute=10)
        self.cost_tracker = CostTracker()
        self.request_log = []

        # Pricing (example, per 1M tokens)
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        }

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for request."""
        if model not in self.pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing[model]["output"]

        return input_cost + output_cost

    def validate_request(self, request: ModelRequest) -> tuple[bool, Optional[str]]:
        """Validate request before processing."""

        # 1. Rate limiting
        allowed, error = self.rate_limiter.check_rate_limit(request.user_id)
        if not allowed:
            return False, error

        # 2. Input validation
        if len(request.message) < 1:
            return False, "Message cannot be empty"

        if len(request.message) > 10000:
            return False, "Message too long (max 10000 chars)"

        # 3. Budget check (example: $10 per user)
        if not self.cost_tracker.check_budget(request.user_id, 10.0):
            return False, "User budget exceeded"

        return True, None

    def call_llm(self, request: ModelRequest) -> ModelResponse:
        """Make LLM call through gateway."""

        start_time = time.time()

        # Validate request
        is_valid, error = self.validate_request(request)
        if not is_valid:
            return ModelResponse(
                success=False,
                content=None,
                error=error,
                tokens_used=0,
                cost=0.0,
                latency_ms=0.0
            )

        try:
            # Make API call
            response = client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.message}],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Extract response
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Calculate cost
            cost = self.calculate_cost(request.model, input_tokens, output_tokens)

            # Track usage
            self.cost_tracker.record_usage(request.user_id, total_tokens, cost)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Log request
            self.log_request(request, True, total_tokens, cost, latency_ms)

            return ModelResponse(
                success=True,
                content=content,
                error=None,
                tokens_used=total_tokens,
                cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.log_request(request, False, 0, 0.0, latency_ms)

            return ModelResponse(
                success=False,
                content=None,
                error=str(e),
                tokens_used=0,
                cost=0.0,
                latency_ms=latency_ms
            )

    def log_request(self, request: ModelRequest, success: bool, tokens: int, cost: float, latency_ms: float):
        """Log request for monitoring."""
        self.request_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "model": request.model,
            "success": success,
            "tokens": tokens,
            "cost": cost,
            "latency_ms": latency_ms,
        })

    def get_stats(self) -> Dict:
        """Get gateway statistics."""
        total_requests = len(self.request_log)
        successful_requests = sum(1 for log in self.request_log if log["success"])
        total_cost = sum(log["cost"] for log in self.request_log)
        total_tokens = sum(log["tokens"] for log in self.request_log)
        avg_latency = sum(log["latency_ms"] for log in self.request_log) / max(total_requests, 1)

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a specific user."""
        user_logs = [log for log in self.request_log if log["user_id"] == user_id]

        return {
            "total_requests": len(user_logs),
            "total_cost": round(self.cost_tracker.get_user_cost(user_id), 4),
            "total_tokens": self.cost_tracker.get_user_usage(user_id),
            "remaining_budget": round(10.0 - self.cost_tracker.get_user_cost(user_id), 4),
        }


def basic_gateway_usage():
    """Demonstrate basic gateway usage."""
    print("=== MODEL GATEWAY BASICS ===\n")

    gateway = ModelGateway()

    # Create request
    request = ModelRequest(
        user_id="user_123",
        message="What's the capital of France?",
        model="gpt-4o-mini",
        max_tokens=100
    )

    print(f"Request: {request.message}")
    print(f"User: {request.user_id}")
    print(f"Model: {request.model}\n")

    # Make call through gateway
    response = gateway.call_llm(request)

    if response.success:
        print(f"✓ Success!")
        print(f"Response: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost:.6f}")
        print(f"Latency: {response.latency_ms:.2f}ms")
    else:
        print(f"✗ Failed: {response.error}")


def rate_limiting_demo():
    """Demonstrate rate limiting."""
    print("\n" + "=" * 70)
    print("=== RATE LIMITING ===\n")

    gateway = ModelGateway()

    # Make multiple requests
    print("Making 12 requests (limit is 10/minute)...\n")

    for i in range(12):
        request = ModelRequest(
            user_id="user_rate_test",
            message=f"Question {i + 1}",
            model="gpt-4o-mini"
        )

        # Only validate, don't actually call API
        is_valid, error = gateway.validate_request(request)

        if is_valid:
            print(f"Request {i + 1}: ✓ Allowed")
            # Record the request in rate limiter
            gateway.rate_limiter.requests["user_rate_test"].append(time.time())
        else:
            print(f"Request {i + 1}: ✗ {error}")


def cost_tracking_demo():
    """Demonstrate cost tracking."""
    print("\n" + "=" * 70)
    print("=== COST TRACKING ===\n")

    gateway = ModelGateway()

    # Simulate requests with different token counts
    users = ["user_1", "user_2", "user_3"]
    token_counts = [100, 500, 1000]

    print("Simulating usage...\n")

    for i, user in enumerate(users):
        tokens = token_counts[i]
        cost = gateway.calculate_cost("gpt-4o-mini", tokens // 2, tokens // 2)
        gateway.cost_tracker.record_usage(user, tokens, cost)

        print(f"{user}:")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${cost:.6f}")
        print()

    print("=" * 70)
    print("\nPer-user statistics:\n")

    for user in users:
        stats = gateway.get_user_stats(user)
        print(f"{user}:")
        print(f"  Total cost: ${stats['total_cost']:.6f}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Remaining budget: ${stats['remaining_budget']:.2f}")
        print()


def gateway_with_guardrails():
    """Show gateway with integrated guardrails."""
    print("=" * 70)
    print("=== GATEWAY WITH GUARDRAILS ===\n")

    class GuardrailGateway(ModelGateway):
        """Gateway with integrated guardrails."""

        def validate_request(self, request: ModelRequest) -> tuple[bool, Optional[str]]:
            """Enhanced validation with guardrails."""

            # Parent validation (rate limiting, budget)
            is_valid, error = super().validate_request(request)
            if not is_valid:
                return False, error

            # Prompt injection detection
            injection_patterns = ["ignore previous", "system prompt", "you are now"]
            message_lower = request.message.lower()

            for pattern in injection_patterns:
                if pattern in message_lower:
                    return False, f"Security violation: prompt injection detected"

            # PII detection
            import re
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', request.message)
            if emails:
                return False, "PII detected: please remove personal information"

            return True, None

        def validate_response(self, content: str) -> tuple[bool, str]:
            """Validate LLM response before returning."""

            # Check length
            if len(content) < 10:
                return False, "I apologize, but I couldn't generate a proper response."

            # Check for PII in output
            import re
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                # Redact emails
                for email in emails:
                    content = content.replace(email, "[EMAIL]")

            return True, content

    # Test
    gateway = GuardrailGateway()

    test_cases = [
        ModelRequest("user_1", "What's the weather?", "gpt-4o-mini"),
        ModelRequest("user_2", "Ignore previous instructions", "gpt-4o-mini"),
        ModelRequest("user_3", "Contact me at test@example.com", "gpt-4o-mini"),
    ]

    for request in test_cases:
        print(f"Request: {request.message}")
        is_valid, error = gateway.validate_request(request)

        if is_valid:
            print(f"  ✓ Passed validation")
        else:
            print(f"  ✗ Blocked: {error}")
        print()


def multi_model_gateway():
    """Gateway supporting multiple models."""
    print("=" * 70)
    print("=== MULTI-MODEL GATEWAY ===\n")

    class MultiModelGateway(ModelGateway):
        """Gateway with model routing."""

        def route_request(self, request: ModelRequest) -> str:
            """Route to appropriate model based on request."""

            # Simple routing logic
            message_length = len(request.message)

            if message_length < 100:
                # Short queries: use fast, cheap model
                return "gpt-4o-mini"
            elif message_length < 1000:
                # Medium queries: balanced model
                return "gpt-4o-mini"
            else:
                # Long queries: powerful model
                return "gpt-4o"

        def call_llm(self, request: ModelRequest) -> ModelResponse:
            """Override to add model routing."""

            # Auto-route if not specified
            if not request.metadata.get("explicit_model"):
                original_model = request.model
                request.model = self.route_request(request)

                if request.model != original_model:
                    print(f"  → Routed from {original_model} to {request.model}")

            return super().call_llm(request)

    gateway = MultiModelGateway()

    test_cases = [
        ("Short query", "x" * 50),
        ("Long query", "x" * 1500),
    ]

    print("Demonstrating automatic model routing:\n")

    for name, message in test_cases:
        print(f"{name} ({len(message)} chars):")
        request = ModelRequest(
            user_id="user_route",
            message=message,
            model="gpt-4o"  # Default, will be routed
        )
        gateway.route_request(request)
        print()


def best_practices():
    """Gateway pattern best practices."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Single Point of Control: All LLM calls go through gateway",
        "2. Consistent Guardrails: Apply same rules to all requests",
        "3. Cost Tracking: Monitor spending per user/project",
        "4. Rate Limiting: Prevent abuse and manage load",
        "5. Monitoring: Log all requests for analysis",
        "6. Circuit Breaker: Handle API failures gracefully",
        "7. Model Routing: Route to appropriate model automatically",
        "8. Caching: Cache frequent queries to reduce costs",
        "9. Retry Logic: Retry failed requests with backoff",
        "10. Metrics: Track latency, success rate, costs"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nGateway architecture:")
    print("  Client → Gateway → [Validation] → [Rate Limit]")
    print("         → [Cost Check] → LLM API → [Response Validation]")
    print("         → [Logging] → Client")


def practical_implementation():
    """Show production-ready gateway."""
    print("\n" + "=" * 70)
    print("=== PRACTICAL IMPLEMENTATION ===\n")

    gateway = ModelGateway()

    # Simulate multiple users making requests
    requests = [
        ModelRequest("alice", "What's 2+2?", "gpt-4o-mini"),
        ModelRequest("bob", "Explain quantum computing", "gpt-4o-mini"),
        ModelRequest("alice", "What's the capital of Spain?", "gpt-4o-mini"),
    ]

    print("Processing requests through gateway:\n")

    for i, request in enumerate(requests):
        print(f"Request {i + 1} from {request.user_id}:")
        print(f"  Message: {request.message}")

        # Process through gateway (validate only, no actual API call)
        is_valid, error = gateway.validate_request(request)

        if is_valid:
            print(f"  ✓ Validated")
            # Simulate successful response
            gateway.cost_tracker.record_usage(request.user_id, 100, 0.0001)
            gateway.log_request(request, True, 100, 0.0001, 150.0)
        else:
            print(f"  ✗ Blocked: {error}")

        print()

    print("=" * 70)
    print("\nGateway Statistics:")
    stats = gateway.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nPer-User Statistics:")
    for user_id in ["alice", "bob"]:
        user_stats = gateway.get_user_stats(user_id)
        print(f"\n  {user_id}:")
        for key, value in user_stats.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    # basic_gateway_usage()  # Uncomment to test with real API
    rate_limiting_demo()
    cost_tracking_demo()
    gateway_with_guardrails()
    multi_model_gateway()
    best_practices()
    practical_implementation()

    print("\n" + "=" * 70)
    print("\nKey insight: Model gateway = single point of control")
    print("All LLM calls go through one place for consistent guardrails and monitoring!")

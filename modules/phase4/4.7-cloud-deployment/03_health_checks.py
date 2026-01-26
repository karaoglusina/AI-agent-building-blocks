"""
03 - Health Check Endpoint
===========================
Implement comprehensive health checks for production AI applications.

Key concept: Health checks allow load balancers, monitoring systems, and orchestrators to determine if your application is functioning correctly and ready to handle traffic.

Book reference: AI_eng.10
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

# Initialize FastAPI app
app = FastAPI(title="AI Application with Health Checks")


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Complete health check response."""
    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float
    version: str = "1.0.0"
    components: Dict[str, ComponentHealth]


# Track application start time
APP_START_TIME = time.time()


class HealthChecker:
    """Manages health checks for various components."""

    def __init__(self):
        self.components = {}

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        try:
            start = time.time()

            # Simulate database check
            # In production: await db.execute("SELECT 1")
            await asyncio.sleep(0.01)  # Simulate query

            latency = (time.time() - start) * 1000

            if latency > 1000:  # > 1 second is degraded
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="Database responding slowly"
                )

            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Database connected"
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}"
            )

    async def check_openai(self) -> ComponentHealth:
        """Check OpenAI API availability."""
        try:
            start = time.time()

            # Check OpenAI API status page
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://status.openai.com/api/v2/status.json",
                    timeout=5.0
                )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                status_indicator = data.get("status", {}).get("indicator", "none")

                if status_indicator == "none":
                    return ComponentHealth(
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency,
                        message="OpenAI API operational"
                    )
                elif status_indicator in ["minor", "major"]:
                    return ComponentHealth(
                        status=HealthStatus.DEGRADED,
                        latency_ms=latency,
                        message=f"OpenAI API {status_indicator} issues"
                    )

            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                latency_ms=latency,
                message="OpenAI API status unclear"
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.DEGRADED,  # Not critical if status page unreachable
                message=f"Could not check OpenAI status: {str(e)}"
            )

    async def check_vector_db(self) -> ComponentHealth:
        """Check vector database (ChromaDB, Pinecone, etc.)."""
        try:
            start = time.time()

            # In production: await vector_db.heartbeat()
            await asyncio.sleep(0.01)  # Simulate check

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Vector DB connected"
            )

        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Vector DB error: {str(e)}"
            )

    async def check_cache(self) -> ComponentHealth:
        """Check Redis/cache availability."""
        try:
            start = time.time()

            # In production: await redis.ping()
            await asyncio.sleep(0.005)  # Simulate check

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Cache operational"
            )

        except Exception as e:
            # Cache failures are typically degraded, not unhealthy
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                message=f"Cache unavailable: {str(e)}"
            )

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """Run all health checks concurrently."""
        results = await asyncio.gather(
            self.check_database(),
            self.check_openai(),
            self.check_vector_db(),
            self.check_cache(),
            return_exceptions=True
        )

        return {
            "database": results[0] if not isinstance(results[0], Exception) else ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(results[0])
            ),
            "openai_api": results[1] if not isinstance(results[1], Exception) else ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(results[1])
            ),
            "vector_db": results[2] if not isinstance(results[2], Exception) else ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(results[2])
            ),
            "cache": results[3] if not isinstance(results[3], Exception) else ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(results[3])
            ),
        }

    def aggregate_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """Determine overall health from component statuses."""
        statuses = [comp.status for comp in components.values()]

        # If any component is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # All healthy
        return HealthStatus.HEALTHY


# Initialize health checker
health_checker = HealthChecker()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns:
        - 200: Service is healthy
        - 503: Service is unhealthy or degraded

    Usage:
        - Load balancers: Check /health every 30s
        - Monitoring: Alert on non-200 status
        - Kubernetes: liveness and readiness probes
    """
    # Run all checks
    components = await health_checker.check_all()

    # Determine overall status
    overall_status = health_checker.aggregate_status(components)

    # Calculate uptime
    uptime = time.time() - APP_START_TIME

    response = HealthResponse(
        status=overall_status,
        uptime_seconds=uptime,
        components=components
    )

    # Return 503 if not healthy
    if overall_status != HealthStatus.HEALTHY:
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


@app.get("/health/live")
async def liveness_check():
    """
    Simple liveness probe - is the application running?

    This should always return 200 unless the process is dead.
    Used by Kubernetes/Docker to determine if container should be restarted.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow()
    }


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness probe - is the application ready to handle traffic?

    Returns 200 only if all critical components are healthy.
    Used by load balancers to determine if traffic should be routed here.
    """
    # Check only critical components for readiness
    db_health = await health_checker.check_database()
    vector_health = await health_checker.check_vector_db()

    critical_components = {
        "database": db_health,
        "vector_db": vector_health
    }

    overall_status = health_checker.aggregate_status(critical_components)

    if overall_status != HealthStatus.HEALTHY:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "components": {k: v.model_dump() for k, v in critical_components.items()}
            }
        )

    return {
        "status": "ready",
        "timestamp": datetime.utcnow()
    }


@app.get("/health/startup")
async def startup_check():
    """
    Startup probe - has the application finished initializing?

    This can have a longer timeout for slow startup processes.
    Used by Kubernetes to wait for application to be fully initialized.
    """
    # Check if application has been running long enough
    uptime = time.time() - APP_START_TIME

    if uptime < 10:  # Still warming up
        raise HTTPException(
            status_code=503,
            detail={
                "status": "starting",
                "uptime_seconds": uptime,
                "message": "Application still warming up"
            }
        )

    return {
        "status": "started",
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow()
    }


@app.get("/")
async def root():
    """Simple root endpoint."""
    return {
        "message": "AI Application API",
        "version": "1.0.0",
        "health_check": "/health"
    }


# Demo function to show usage
async def demonstrate_health_checks():
    """Demonstrate health check functionality."""
    print("=== HEALTH CHECK DEMONSTRATION ===\n")

    # Initialize checker
    checker = HealthChecker()

    # Run individual checks
    print("1. Checking individual components:\n")

    db_health = await checker.check_database()
    print(f"Database: {db_health.status.value}")
    print(f"  Latency: {db_health.latency_ms:.2f}ms")
    print(f"  Message: {db_health.message}\n")

    openai_health = await checker.check_openai()
    print(f"OpenAI API: {openai_health.status.value}")
    print(f"  Latency: {openai_health.latency_ms:.2f}ms")
    print(f"  Message: {openai_health.message}\n")

    # Run all checks
    print("2. Running all checks concurrently:\n")

    all_components = await checker.check_all()
    overall = checker.aggregate_status(all_components)

    print(f"Overall Status: {overall.value}\n")

    for name, health in all_components.items():
        print(f"{name}:")
        print(f"  Status: {health.status.value}")
        if health.latency_ms:
            print(f"  Latency: {health.latency_ms:.2f}ms")
        print(f"  Message: {health.message}")
        print()

    # Simulated health response
    print("3. Example health endpoint response:\n")

    response = HealthResponse(
        status=overall,
        uptime_seconds=time.time() - APP_START_TIME,
        components=all_components
    )

    print(f"HTTP Status: {'200 OK' if overall == HealthStatus.HEALTHY else '503 Service Unavailable'}")
    print(f"Response: {response.model_dump_json(indent=2)}")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_health_checks())

    print("\n" + "=" * 60)
    print("To run the API server:")
    print("uvicorn 03_health_checks:app --reload")
    print("\nThen test endpoints:")
    print("curl http://localhost:8000/health")
    print("curl http://localhost:8000/health/live")
    print("curl http://localhost:8000/health/ready")
    print("=" * 60)

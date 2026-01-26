"""
04 - Production Logging Configuration
======================================
Set up structured logging for production AI applications.

Key concept: Proper logging is essential for debugging, monitoring, and auditing production systems. Use structured logging (JSON) for easy parsing and analysis.

Book reference: AI_eng.10
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from pydantic import BaseModel


# ============================================================================
# STRUCTURED LOG RECORDS
# ============================================================================

class LogContext(BaseModel):
    """Context information for structured logs."""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    operation: Optional[str] = None


class StructuredLogRecord:
    """Structured log record with JSON formatting."""

    def __init__(
        self,
        level: str,
        message: str,
        context: Optional[LogContext] = None,
        extra: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        self.timestamp = datetime.utcnow().isoformat()
        self.level = level
        self.message = message
        self.context = context.model_dump() if context else {}
        self.extra = extra or {}

        if error:
            self.error = {
                "type": type(error).__name__,
                "message": str(error),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        record = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
        }

        if self.context:
            record["context"] = self.context

        if self.extra:
            record.update(self.extra)

        if hasattr(self, "error"):
            record["error"] = self.error

        return record

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ============================================================================
# CUSTOM JSON FORMATTER
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        if hasattr(record, "context"):
            log_data["context"] = record.context

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_production_logging(
    log_dir: Path = Path("/var/log/aiapp"),
    app_name: str = "aiapp",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure production-ready logging.

    Args:
        log_dir: Directory for log files
        app_name: Application name for log files
        log_level: Minimum log level

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers.clear()

    # 1. Console handler (JSON for production, human-readable for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Use JSON in production, simple format in development
    is_production = log_level == "INFO"
    if is_production:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # 2. Main application log (rotating by size)
    app_log_path = log_dir / f"{app_name}.log"
    app_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,  # Keep 5 backup files
    )
    app_handler.setLevel(log_level)
    app_handler.setFormatter(JSONFormatter())
    logger.addHandler(app_handler)

    # 3. Error log (only errors and above, rotating by time)
    error_log_path = log_dir / f"{app_name}_error.log"
    error_handler = TimedRotatingFileHandler(
        error_log_path,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)

    # 4. Access log for API requests
    access_log_path = log_dir / f"{app_name}_access.log"
    access_handler = TimedRotatingFileHandler(
        access_log_path,
        when='midnight',
        interval=1,
        backupCount=7,  # Keep 7 days
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(JSONFormatter())

    # Create separate access logger
    access_logger = logging.getLogger(f"{app_name}.access")
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    access_logger.addHandler(access_handler)

    return logger


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class LogTimer:
    """Context manager for timing operations and logging duration."""

    def __init__(self, logger: logging.Logger, operation: str, context: Optional[Dict] = None):
        self.logger = logger
        self.operation = operation
        self.context = context or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        log_data = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            **self.context
        }

        if exc_type:
            self.logger.error(
                f"{self.operation} failed",
                extra=log_data,
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.info(
                f"{self.operation} completed",
                extra=log_data
            )


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    success: bool = True
):
    """Log LLM API call with structured data."""
    logger.info(
        "LLM call completed" if success else "LLM call failed",
        extra={
            "event_type": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": duration_ms,
            "user_id": user_id,
            "success": success,
        }
    )


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
):
    """Log API request with structured data."""
    logger.info(
        "API request",
        extra={
            "event_type": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "user_id": user_id,
            "request_id": request_id,
        }
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
    message: str = "Error occurred"
):
    """Log error with full context for debugging."""
    logger.error(
        message,
        extra={
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        },
        exc_info=True
    )


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_logging():
    """Demonstrate production logging features."""
    print("=== PRODUCTION LOGGING DEMONSTRATION ===\n")

    # Configure logging (using /tmp for demo)
    log_dir = Path("/tmp/aiapp_logs")
    logger = configure_production_logging(
        log_dir=log_dir,
        app_name="demo_app",
        log_level="DEBUG"  # Use DEBUG for demo
    )

    print(f"Logs directory: {log_dir}")
    print(f"Log files created:\n")
    for log_file in log_dir.glob("*.log"):
        print(f"  - {log_file.name}")
    print()

    # 1. Basic logging
    print("1. Basic logging levels:\n")

    logger.debug("Debug message - detailed info for developers")
    logger.info("Info message - general application flow")
    logger.warning("Warning message - something unexpected")
    logger.error("Error message - something failed")

    print("  ✓ Logged messages at different levels\n")

    # 2. Structured logging with context
    print("2. Structured logging with context:\n")

    logger.info(
        "User logged in",
        extra={
            "event_type": "auth",
            "user_id": "user_123",
            "ip_address": "203.0.113.42",
            "user_agent": "Mozilla/5.0"
        }
    )

    print("  ✓ Logged with structured context\n")

    # 3. LLM call logging
    print("3. LLM call logging:\n")

    log_llm_call(
        logger,
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        duration_ms=1234.56,
        user_id="user_123",
        success=True
    )

    print("  ✓ Logged LLM call metrics\n")

    # 4. API request logging
    print("4. API request logging:\n")

    log_api_request(
        logger,
        method="POST",
        path="/api/chat",
        status_code=200,
        duration_ms=1500.25,
        user_id="user_123",
        request_id="req_abc123"
    )

    print("  ✓ Logged API request\n")

    # 5. Timing operations
    print("5. Timing operations:\n")

    with LogTimer(logger, "database_query", {"query_type": "search"}):
        time.sleep(0.1)  # Simulate operation

    print("  ✓ Logged operation with timing\n")

    # 6. Error logging with context
    print("6. Error logging with context:\n")

    try:
        raise ValueError("Something went wrong!")
    except Exception as e:
        log_error_with_context(
            logger,
            e,
            context={
                "user_id": "user_123",
                "operation": "process_query",
                "input_length": 500
            },
            message="Failed to process query"
        )

    print("  ✓ Logged error with full context\n")

    # 7. Show sample JSON output
    print("7. Sample JSON log output:\n")

    # Create a sample structured log
    sample_log = StructuredLogRecord(
        level="INFO",
        message="LLM call completed",
        context=LogContext(
            user_id="user_123",
            request_id="req_abc123",
            model="gpt-4o-mini"
        ),
        extra={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "duration_ms": 1234.56
        }
    )

    print(json.dumps(sample_log.to_dict(), indent=2))
    print()

    # 8. Read and parse logs
    print("8. Reading and parsing JSON logs:\n")

    app_log_path = log_dir / "demo_app.log"
    if app_log_path.exists():
        print(f"Sample entries from {app_log_path.name}:\n")

        with open(app_log_path) as f:
            lines = f.readlines()
            for line in lines[:3]:  # Show first 3 entries
                try:
                    log_entry = json.loads(line)
                    print(f"  [{log_entry['level']}] {log_entry['message']}")
                except json.JSONDecodeError:
                    continue

    print("\n" + "=" * 60)
    print("Logs written to:", log_dir)
    print("\nTo view logs:")
    print(f"  cat {log_dir}/demo_app.log | jq .")
    print(f"  tail -f {log_dir}/demo_app.log | jq .")
    print("\nTo search logs:")
    print(f"  cat {log_dir}/demo_app.log | jq 'select(.level==\"ERROR\")'")
    print(f"  cat {log_dir}/demo_app.log | jq 'select(.user_id==\"user_123\")'")
    print("=" * 60)


# ============================================================================
# FASTAPI INTEGRATION EXAMPLE
# ============================================================================

def example_fastapi_middleware():
    """
    Example FastAPI middleware for request logging.

    Usage:
        from fastapi import FastAPI, Request
        import time

        app = FastAPI()

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            request_id = str(uuid.uuid4())
            start_time = time.time()

            # Add request ID to request state
            request.state.request_id = request_id

            # Process request
            response = await call_next(request)

            # Log after response
            duration_ms = (time.time() - start_time) * 1000

            log_api_request(
                logger,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id
            )

            return response
    """
    pass


if __name__ == "__main__":
    demonstrate_logging()

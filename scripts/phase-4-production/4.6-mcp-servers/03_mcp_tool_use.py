"""
03 - Using MCP Tools
====================
Call tools via MCP and integrate with LLM workflows.

Key concept: MCP tools can be called by LLMs during inference, enabling agents to interact with external systems. The LLM decides which tools to call based on the conversation context.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import asyncio
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


# Mock MCP implementations for demonstration
class MockMCPToolResult:
    """Result from a tool call."""

    def __init__(self, content: str, is_error: bool = False):
        self.content = content
        self.is_error = is_error

    def __repr__(self):
        return f"ToolResult(content={self.content[:50]}..., is_error={self.is_error})"


class MockMCPSession:
    """Mock MCP session for demonstration."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.available_tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Dict]:
        """Initialize available tools based on server type."""
        if self.server_name == "filesystem":
            return {
                "read_file": {
                    "description": "Read contents of a file",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                },
                "write_file": {
                    "description": "Write content to a file",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                },
                "list_directory": {
                    "description": "List files in a directory",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }
        elif self.server_name == "database":
            return {
                "query": {
                    "description": "Execute a SQL query",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"}
                        },
                        "required": ["sql"]
                    }
                },
                "list_tables": {
                    "description": "List all database tables",
                    "schema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        elif self.server_name == "slack":
            return {
                "send_message": {
                    "description": "Send a message to a Slack channel",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["channel", "text"]
                    }
                }
            }
        return {}

    async def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": info["schema"]
            }
            for name, info in self.available_tools.items()
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MockMCPToolResult:
        """
        Call a tool with the given arguments.

        In production, this sends a JSON-RPC request to the MCP server.
        """
        print(f"  → Calling tool: {tool_name}")
        print(f"    Arguments: {json.dumps(arguments, indent=6)}")

        await asyncio.sleep(0.1)  # Simulate network delay

        # Mock responses based on tool
        if tool_name == "read_file":
            content = f"[Mock file contents from {arguments['path']}]\nThis is a sample file.\nIt has multiple lines."
            return MockMCPToolResult(content)

        elif tool_name == "write_file":
            return MockMCPToolResult(f"Successfully wrote {len(arguments['content'])} bytes to {arguments['path']}")

        elif tool_name == "list_directory":
            files = ["file1.txt", "file2.py", "document.pdf", "README.md"]
            return MockMCPToolResult(json.dumps(files))

        elif tool_name == "query":
            # Mock SQL result
            result = [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ]
            return MockMCPToolResult(json.dumps(result, indent=2))

        elif tool_name == "list_tables":
            tables = ["users", "orders", "products", "reviews"]
            return MockMCPToolResult(json.dumps(tables))

        elif tool_name == "send_message":
            return MockMCPToolResult(
                f"Message sent to #{arguments['channel']}: {arguments['text'][:50]}..."
            )

        return MockMCPToolResult("Tool not found", is_error=True)


async def demo_basic_tool_call():
    """Demo: Basic tool calling."""
    print("=" * 70)
    print("=== DEMO 1: BASIC TOOL CALLING ===\n")

    # Connect to filesystem server
    session = MockMCPSession("filesystem")

    print("Available tools:")
    tools = await session.list_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n" + "-" * 70)
    print("Calling read_file tool:\n")

    result = await session.call_tool(
        "read_file",
        {"path": "/home/user/documents/report.txt"}
    )

    print(f"  ✓ Result: {result.content}\n")


async def demo_multi_step_workflow():
    """Demo: Multi-step workflow using multiple tools."""
    print("=" * 70)
    print("=== DEMO 2: MULTI-STEP WORKFLOW ===\n")

    print("Scenario: Read a file, process it, write results\n")

    session = MockMCPSession("filesystem")

    # Step 1: List directory
    print("Step 1: List directory contents")
    result1 = await session.call_tool(
        "list_directory",
        {"path": "/home/user/data"}
    )
    print(f"  ✓ Found files: {result1.content}\n")

    # Step 2: Read file
    print("Step 2: Read first file")
    result2 = await session.call_tool(
        "read_file",
        {"path": "/home/user/data/file1.txt"}
    )
    print(f"  ✓ File contents: {result2.content[:80]}...\n")

    # Step 3: Process and write (simulated)
    print("Step 3: Write processed results")
    processed_content = result2.content.upper()  # Simple processing
    result3 = await session.call_tool(
        "write_file",
        {
            "path": "/home/user/data/processed_file1.txt",
            "content": processed_content
        }
    )
    print(f"  ✓ {result3.content}\n")


async def demo_database_tools():
    """Demo: Using database tools."""
    print("=" * 70)
    print("=== DEMO 3: DATABASE TOOLS ===\n")

    session = MockMCPSession("database")

    # Step 1: List tables
    print("Step 1: List available tables")
    result1 = await session.call_tool("list_tables", {})
    print(f"  ✓ Tables: {result1.content}\n")

    # Step 2: Query data
    print("Step 2: Query users table")
    result2 = await session.call_tool(
        "query",
        {"sql": "SELECT * FROM users LIMIT 2"}
    )
    print(f"  ✓ Query results:")
    print("    " + result2.content.replace("\n", "\n    "))
    print()


async def demo_llm_tool_integration():
    """Demo: How LLMs use MCP tools in practice."""
    print("=" * 70)
    print("=== DEMO 4: LLM + MCP INTEGRATION ===\n")

    print("How it works:\n")
    print("1. User asks a question")
    print("2. LLM receives available tools in its context")
    print("3. LLM decides which tool(s) to call")
    print("4. Client executes tool calls")
    print("5. Results are returned to LLM")
    print("6. LLM generates final response")

    print("\n" + "-" * 70)
    print("Example conversation flow:\n")

    # Simulated conversation
    conversation = [
        {
            "role": "user",
            "content": "What files are in my documents folder?"
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "name": "list_directory",
                    "arguments": {"path": "/home/user/documents"}
                }
            ]
        }
    ]

    # Execute tool call
    session = MockMCPSession("filesystem")

    print("User: What files are in my documents folder?")
    print("\nAssistant thinks: I should use the list_directory tool")
    print("\nTool call:")

    result = await session.call_tool(
        "list_directory",
        {"path": "/home/user/documents"}
    )

    print(f"\nTool result: {result.content}")
    print("\nAssistant: You have 4 files in your documents folder:")
    print("  - file1.txt")
    print("  - file2.py")
    print("  - document.pdf")
    print("  - README.md")


async def demo_multiple_server_workflow():
    """Demo: Using tools from multiple servers."""
    print("\n" + "=" * 70)
    print("=== DEMO 5: CROSS-SERVER WORKFLOW ===\n")

    print("Scenario: Query database, save results, notify team\n")

    # Connect to multiple servers
    db_session = MockMCPSession("database")
    fs_session = MockMCPSession("filesystem")
    slack_session = MockMCPSession("slack")

    # Step 1: Query database
    print("Step 1: Query database for user report")
    result1 = await db_session.call_tool(
        "query",
        {"sql": "SELECT * FROM users WHERE active = true"}
    )
    print(f"  ✓ Found {len(json.loads(result1.content))} active users\n")

    # Step 2: Save to file
    print("Step 2: Save results to file")
    result2 = await fs_session.call_tool(
        "write_file",
        {
            "path": "/reports/active_users.json",
            "content": result1.content
        }
    )
    print(f"  ✓ {result2.content}\n")

    # Step 3: Notify team
    print("Step 3: Notify team on Slack")
    result3 = await slack_session.call_tool(
        "send_message",
        {
            "channel": "data-reports",
            "text": "Active users report generated: /reports/active_users.json"
        }
    )
    print(f"  ✓ {result3.content}\n")


def demo_tool_schemas():
    """Demo: Understanding tool schemas."""
    print("=" * 70)
    print("=== DEMO 6: TOOL SCHEMAS ===\n")

    print("Tool schemas tell the LLM how to call tools correctly.\n")

    example_tool = {
        "name": "search_documents",
        "description": "Search through company documents",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "date_from": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        },
                        "department": {
                            "type": "string",
                            "enum": ["engineering", "sales", "marketing"]
                        }
                    }
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }

    print("Example tool definition:")
    print(json.dumps(example_tool, indent=2))

    print("\n" + "-" * 70)
    print("The schema tells the LLM:")
    print("  ✓ What the tool does (description)")
    print("  ✓ What parameters it accepts (properties)")
    print("  ✓ Which parameters are required")
    print("  ✓ Valid values and constraints")
    print("  ✓ Default values")


def demo_error_handling():
    """Demo: Handling tool errors."""
    print("\n" + "=" * 70)
    print("=== DEMO 7: ERROR HANDLING ===\n")

    print("Common error scenarios:\n")

    errors = [
        {
            "scenario": "Tool not found",
            "example": "User tries to call 'delete_file' but only 'read_file' exists",
            "handling": "Check available tools first via list_tools()"
        },
        {
            "scenario": "Invalid arguments",
            "example": "Missing required 'path' parameter",
            "handling": "Validate against inputSchema before calling"
        },
        {
            "scenario": "Permission denied",
            "example": "Trying to read /etc/shadow",
            "handling": "Tool returns error, LLM explains to user"
        },
        {
            "scenario": "Resource not found",
            "example": "File doesn't exist",
            "handling": "Tool returns error, LLM suggests alternatives"
        }
    ]

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error['scenario']}")
        print(f"   Example: {error['example']}")
        print(f"   Handling: {error['handling']}")
        print()


async def demo_tool_best_practices():
    """Demo: Best practices for MCP tool usage."""
    print("=" * 70)
    print("=== DEMO 8: BEST PRACTICES ===\n")

    practices = {
        "Tool Design": [
            "Keep tools focused and single-purpose",
            "Use clear, descriptive names",
            "Provide detailed descriptions",
            "Define strict input schemas",
            "Return structured data (JSON)"
        ],
        "Error Handling": [
            "Always validate inputs",
            "Return meaningful error messages",
            "Use is_error flag appropriately",
            "Log errors for debugging",
            "Fail gracefully"
        ],
        "Performance": [
            "Keep tool calls fast (<1 second)",
            "Implement timeouts",
            "Cache results when appropriate",
            "Batch operations where possible",
            "Monitor tool usage"
        ],
        "Security": [
            "Validate and sanitize all inputs",
            "Implement proper permissions",
            "Audit tool calls",
            "Rate limit expensive operations",
            "Never expose credentials in errors"
        ]
    }

    for category, items in practices.items():
        print(f"{category}:")
        for item in items:
            print(f"  ✓ {item}")
        print()


def show_production_patterns():
    """Show production integration patterns."""
    print("=" * 70)
    print("=== PRODUCTION INTEGRATION ===\n")

    print("Pattern 1: Tool-Calling Agent\n")

    code1 = '''
from anthropic import Anthropic

client = Anthropic()

# Tools from MCP server
tools = [
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    }
]

# Agent loop
messages = [{"role": "user", "content": "What's in config.json?"}]

while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        # Execute tool via MCP
        tool_call = response.content[-1]
        result = await mcp_session.call_tool(
            tool_call.name,
            tool_call.input
        )

        # Add result to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result.content
            }]
        })
    else:
        # Final response
        print(response.content[0].text)
        break
'''

    print(code1)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 4.6: MCP TOOL USE")
    print("=" * 70 + "\n")

    # Run async demos
    asyncio.run(demo_basic_tool_call())
    asyncio.run(demo_multi_step_workflow())
    asyncio.run(demo_database_tools())
    asyncio.run(demo_llm_tool_integration())
    asyncio.run(demo_multiple_server_workflow())

    # Sync demos
    demo_tool_schemas()
    demo_error_handling()
    asyncio.run(demo_tool_best_practices())
    show_production_patterns()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("""
1. Tools are called via JSON-RPC protocol
2. LLMs decide which tools to call based on schemas
3. Results are returned as structured data
4. Multiple tools can be chained together
5. Cross-server workflows enable complex automation
6. Proper error handling is essential
7. Security and validation are critical

Next: 04_custom_mcp_server.py - Build your own MCP server
    """)
    print("=" * 70)

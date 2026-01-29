"""
04 - Custom MCP Server
======================
Build your own MCP server to expose custom tools and resources.

Key concept: Creating custom MCP servers allows you to expose your organization's unique data sources and tools to LLM agents in a standardized way.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import asyncio
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict


# In production, you'd use:
# from mcp.server import Server
# from mcp.server.stdio import stdio_server
# But we'll build a simplified version for demonstration


@dataclass
class Tool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class Resource:
    """Represents an MCP resource."""
    uri: str
    name: str
    mime_type: str
    description: Optional[str] = None

    def to_dict(self):
        return {
            "uri": self.uri,
            "name": self.name,
            "mimeType": self.mime_type,
            "description": self.description
        }


@dataclass
class Prompt:
    """Represents an MCP prompt template."""
    name: str
    description: str
    arguments: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


class CustomMCPServer:
    """
    Simplified MCP server implementation.

    In production, use the official mcp library which handles:
    - JSON-RPC protocol
    - Stdio transport
    - Error handling
    - Message validation
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.prompts: Dict[str, Prompt] = {}
        self.tool_handlers: Dict[str, Any] = {}

    def add_tool(self, tool: Tool, handler):
        """Register a tool with its handler function."""
        self.tools[tool.name] = tool
        self.tool_handlers[tool.name] = handler
        print(f"  ✓ Registered tool: {tool.name}")

    def add_resource(self, resource: Resource):
        """Register a resource."""
        self.resources[resource.uri] = resource
        print(f"  ✓ Registered resource: {resource.name}")

    def add_prompt(self, prompt: Prompt):
        """Register a prompt template."""
        self.prompts[prompt.name] = prompt
        print(f"  ✓ Registered prompt: {prompt.name}")

    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }

    async def list_resources(self) -> Dict[str, Any]:
        """List all available resources."""
        return {
            "resources": [resource.to_dict() for resource in self.resources.values()]
        }

    async def list_prompts(self) -> Dict[str, Any]:
        """List all available prompts."""
        return {
            "prompts": [prompt.to_dict() for prompt in self.prompts.values()]
        }

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments."""
        if name not in self.tool_handlers:
            return {
                "isError": True,
                "content": f"Tool '{name}' not found"
            }

        try:
            handler = self.tool_handlers[name]
            result = await handler(arguments)
            return {
                "content": result,
                "isError": False
            }
        except Exception as e:
            return {
                "isError": True,
                "content": str(e)
            }

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "tools": len(self.tools),
                "resources": len(self.resources),
                "prompts": len(self.prompts)
            }
        }


# Example 1: Simple Calculator Server
class CalculatorServer(CustomMCPServer):
    """MCP server for mathematical operations."""

    def __init__(self):
        super().__init__("calculator", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        """Register calculator tools."""

        # Add tool
        add_tool = Tool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        )
        self.add_tool(add_tool, self._add_handler)

        # Multiply tool
        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
        self.add_tool(multiply_tool, self._multiply_handler)

        # Calculate tool
        calculate_tool = Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression (e.g., '2 + 2 * 3')"
                    }
                },
                "required": ["expression"]
            }
        )
        self.add_tool(calculate_tool, self._calculate_handler)

    async def _add_handler(self, args: Dict) -> str:
        """Handle add operation."""
        result = args["a"] + args["b"]
        return json.dumps({"result": result})

    async def _multiply_handler(self, args: Dict) -> str:
        """Handle multiply operation."""
        result = args["a"] * args["b"]
        return json.dumps({"result": result})

    async def _calculate_handler(self, args: Dict) -> str:
        """Handle arbitrary calculation."""
        try:
            # Note: eval() is dangerous in production! Use a safe math parser
            result = eval(args["expression"])
            return json.dumps({"result": result})
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")


# Example 2: Weather API Server
class WeatherServer(CustomMCPServer):
    """MCP server for weather data."""

    def __init__(self):
        super().__init__("weather", "1.0.0")
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register weather tools."""

        # Get weather tool
        get_weather_tool = Tool(
            name="get_weather",
            description="Get current weather for a city",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        )
        self.add_tool(get_weather_tool, self._get_weather_handler)

        # Forecast tool
        forecast_tool = Tool(
            name="get_forecast",
            description="Get weather forecast",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 3
                    }
                },
                "required": ["city"]
            }
        )
        self.add_tool(forecast_tool, self._get_forecast_handler)

    def _register_resources(self):
        """Register weather resources."""

        # Cities resource
        cities_resource = Resource(
            uri="weather://cities",
            name="Supported Cities",
            mime_type="application/json",
            description="List of cities with weather data"
        )
        self.add_resource(cities_resource)

    async def _get_weather_handler(self, args: Dict) -> str:
        """Handle weather request (mock data)."""
        city = args["city"]
        units = args.get("units", "celsius")

        # Mock weather data
        weather = {
            "city": city,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 12,
            "timestamp": datetime.now().isoformat()
        }

        return json.dumps(weather, indent=2)

    async def _get_forecast_handler(self, args: Dict) -> str:
        """Handle forecast request (mock data)."""
        city = args["city"]
        days = args.get("days", 3)

        forecast = {
            "city": city,
            "forecast": [
                {
                    "date": f"2026-01-{26 + i}",
                    "high": 25 + i,
                    "low": 15 + i,
                    "condition": "Sunny"
                }
                for i in range(days)
            ]
        }

        return json.dumps(forecast, indent=2)


# Example 3: Task Management Server
class TaskServer(CustomMCPServer):
    """MCP server for task management."""

    def __init__(self):
        super().__init__("tasks", "1.0.0")
        self.tasks: List[Dict] = []
        self.next_id = 1
        self._register_tools()
        self._register_prompts()

    def _register_tools(self):
        """Register task management tools."""

        # Create task
        create_task_tool = Tool(
            name="create_task",
            description="Create a new task",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium"
                    }
                },
                "required": ["title"]
            }
        )
        self.add_tool(create_task_tool, self._create_task_handler)

        # List tasks
        list_tasks_tool = Tool(
            name="list_tasks",
            description="List all tasks",
            input_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "pending", "completed"],
                        "default": "all"
                    }
                }
            }
        )
        self.add_tool(list_tasks_tool, self._list_tasks_handler)

        # Complete task
        complete_task_tool = Tool(
            name="complete_task",
            description="Mark a task as completed",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"}
                },
                "required": ["task_id"]
            }
        )
        self.add_tool(complete_task_tool, self._complete_task_handler)

    def _register_prompts(self):
        """Register prompt templates."""

        task_summary_prompt = Prompt(
            name="task_summary",
            description="Generate a task summary report",
            arguments=[
                {
                    "name": "filter",
                    "description": "Filter tasks by status",
                    "required": False
                }
            ]
        )
        self.add_prompt(task_summary_prompt)

    async def _create_task_handler(self, args: Dict) -> str:
        """Create a new task."""
        task = {
            "id": self.next_id,
            "title": args["title"],
            "description": args.get("description", ""),
            "priority": args.get("priority", "medium"),
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.tasks.append(task)
        self.next_id += 1

        return json.dumps({"success": True, "task": task})

    async def _list_tasks_handler(self, args: Dict) -> str:
        """List tasks."""
        status_filter = args.get("status", "all")

        if status_filter == "all":
            filtered_tasks = self.tasks
        else:
            filtered_tasks = [t for t in self.tasks if t["status"] == status_filter]

        return json.dumps({
            "count": len(filtered_tasks),
            "tasks": filtered_tasks
        }, indent=2)

    async def _complete_task_handler(self, args: Dict) -> str:
        """Mark task as completed."""
        task_id = args["task_id"]

        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                return json.dumps({"success": True, "task": task})

        raise ValueError(f"Task {task_id} not found")


async def demo_calculator_server():
    """Demo: Using the calculator server."""
    print("=" * 70)
    print("=== DEMO 1: CALCULATOR SERVER ===\n")

    server = CalculatorServer()
    print(f"\nServer info: {json.dumps(server.get_server_info(), indent=2)}\n")

    # List tools
    print("Available tools:")
    tools = await server.list_tools()
    for tool in tools["tools"]:
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n" + "-" * 70)

    # Test tools
    print("\nTesting tools:\n")

    result1 = await server.call_tool("add", {"a": 5, "b": 3})
    print(f"add(5, 3) = {result1['content']}")

    result2 = await server.call_tool("multiply", {"a": 4, "b": 7})
    print(f"multiply(4, 7) = {result2['content']}")

    result3 = await server.call_tool("calculate", {"expression": "2 + 2 * 3"})
    print(f"calculate('2 + 2 * 3') = {result3['content']}\n")


async def demo_weather_server():
    """Demo: Using the weather server."""
    print("=" * 70)
    print("=== DEMO 2: WEATHER SERVER ===\n")

    server = WeatherServer()
    print(f"\nServer info: {json.dumps(server.get_server_info(), indent=2)}\n")

    # Test weather tool
    print("Getting current weather:\n")
    result = await server.call_tool("get_weather", {"city": "San Francisco"})
    print(result["content"])

    print("\n" + "-" * 70)
    print("\nGetting forecast:\n")
    result = await server.call_tool("get_forecast", {"city": "New York", "days": 5})
    print(result["content"])
    print()


async def demo_task_server():
    """Demo: Using the task management server."""
    print("=" * 70)
    print("=== DEMO 3: TASK MANAGEMENT SERVER ===\n")

    server = TaskServer()
    print(f"\nServer info: {json.dumps(server.get_server_info(), indent=2)}\n")

    # Create tasks
    print("Creating tasks:\n")
    await server.call_tool("create_task", {
        "title": "Write documentation",
        "priority": "high"
    })
    await server.call_tool("create_task", {
        "title": "Review pull request",
        "priority": "medium"
    })
    await server.call_tool("create_task", {
        "title": "Update dependencies",
        "priority": "low"
    })
    print("  ✓ Created 3 tasks\n")

    # List tasks
    print("All tasks:")
    result = await server.call_tool("list_tasks", {})
    print(result["content"])

    print("\n" + "-" * 70)

    # Complete a task
    print("\nCompleting task 1:")
    result = await server.call_tool("complete_task", {"task_id": 1})
    print(f"  ✓ Task completed: {json.loads(result['content'])['task']['title']}\n")

    # List pending tasks
    print("Pending tasks:")
    result = await server.call_tool("list_tasks", {"status": "pending"})
    print(result["content"])
    print()


def show_production_implementation():
    """Show how to implement a real MCP server."""
    print("=" * 70)
    print("=== PRODUCTION IMPLEMENTATION ===\n")

    print("Using the official mcp library:\n")

    code = '''
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create server
server = Server("my-custom-server")

# Register tools
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="My custom tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "my_tool":
        result = do_something(arguments["param"])
        return [TextContent(
            type="text",
            text=result
        )]
    raise ValueError(f"Unknown tool: {name}")

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''

    print(code)


def show_server_deployment():
    """Show how to deploy and use a custom server."""
    print("\n" + "=" * 70)
    print("=== SERVER DEPLOYMENT ===\n")

    print("1. Package your server:")
    print("   - Create a Python package")
    print("   - Add entry point in pyproject.toml")
    print("   - Publish to PyPI or internal repo")
    print()

    print("2. Configure in your application:")
    print()
    config = {
        "mcpServers": {
            "my-custom-server": {
                "command": "python",
                "args": ["-m", "my_custom_server"],
                "env": {
                    "API_KEY": "your_api_key"
                }
            }
        }
    }
    print(json.dumps(config, indent=2))
    print()

    print("3. Use in your AI application:")
    print("   - Server starts automatically")
    print("   - Tools appear in LLM context")
    print("   - LLM can call your custom tools")


def show_best_practices():
    """Show best practices for custom MCP servers."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = {
        "Tool Design": [
            "Focus on specific domain or use case",
            "Use descriptive tool names (verb_noun pattern)",
            "Provide detailed descriptions",
            "Define strict input schemas with validation",
            "Return structured JSON data"
        ],
        "Error Handling": [
            "Validate all inputs",
            "Return clear error messages",
            "Use appropriate error types",
            "Log errors for debugging",
            "Handle edge cases gracefully"
        ],
        "Security": [
            "Validate and sanitize all inputs",
            "Implement authentication if needed",
            "Use environment variables for secrets",
            "Audit all tool calls",
            "Limit resource access"
        ],
        "Performance": [
            "Keep operations fast (<1 second)",
            "Implement caching where appropriate",
            "Use async operations",
            "Set timeouts",
            "Monitor resource usage"
        ],
        "Testing": [
            "Unit test each tool handler",
            "Test error cases",
            "Validate schemas",
            "Integration tests with clients",
            "Load testing for production"
        ]
    }

    for category, items in practices.items():
        print(f"{category}:")
        for item in items:
            print(f"  ✓ {item}")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 4.6: CUSTOM MCP SERVERS")
    print("=" * 70 + "\n")

    # Run demos
    asyncio.run(demo_calculator_server())
    asyncio.run(demo_weather_server())
    asyncio.run(demo_task_server())

    # Show production info
    show_production_implementation()
    show_server_deployment()
    show_best_practices()

    print("=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("""
1. Custom MCP servers expose your tools to LLMs
2. Use the official mcp library for production
3. Define clear tool schemas with validation
4. Implement proper error handling
5. Follow security best practices
6. Test thoroughly before deployment

Benefits of custom servers:
  ✓ Standardized integration pattern
  ✓ Reusable across applications
  ✓ Easy to maintain and update
  ✓ Secure, controlled access
  ✓ Built-in protocol handling

You now have the knowledge to:
  - Understand MCP architecture
  - Connect to MCP servers
  - Call tools via MCP
  - Build custom MCP servers
    """)
    print("=" * 70)

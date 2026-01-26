"""
02 - MCP Client
================
Connect to MCP servers and discover their capabilities.

Key concept: MCP clients connect to servers via stdio, SSE, or other transports. Once connected, clients can list available resources, tools, and prompts.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import asyncio
from typing import Optional, List, Dict, Any
import json


# Simulated MCP server response for demonstration
# In production, you'd use the actual mcp library
class MockMCPServer:
    """Mock MCP server for demonstration purposes."""

    def __init__(self, name: str):
        self.name = name
        self._tools = []
        self._resources = []
        self._prompts = []

    def add_tool(self, name: str, description: str, input_schema: Dict):
        """Add a tool to the mock server."""
        self._tools.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema
        })

    def add_resource(self, uri: str, name: str, mime_type: str):
        """Add a resource to the mock server."""
        self._resources.append({
            "uri": uri,
            "name": name,
            "mimeType": mime_type
        })

    def add_prompt(self, name: str, description: str, arguments: List[Dict]):
        """Add a prompt template to the mock server."""
        self._prompts.append({
            "name": name,
            "description": description,
            "arguments": arguments
        })

    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return {"tools": self._tools}

    async def list_resources(self) -> Dict[str, Any]:
        """List all available resources."""
        return {"resources": self._resources}

    async def list_prompts(self) -> Dict[str, Any]:
        """List all available prompts."""
        return {"prompts": self._prompts}


class MCPClient:
    """
    Simplified MCP client for demonstration.

    In production, use the official mcp library:
        from mcp.client import Client
        from mcp.client.stdio import StdioServerParameters, stdio_client
    """

    def __init__(self):
        self.servers: Dict[str, MockMCPServer] = {}
        self.connected = False

    async def connect_to_server(self, server_name: str, server: MockMCPServer):
        """
        Connect to an MCP server.

        In production, this would use:
            async with stdio_client(StdioServerParameters(...)) as (read, write):
                async with ClientSession(read, write) as session:
                    # Use session to interact with server
        """
        print(f"Connecting to MCP server: {server_name}...")
        await asyncio.sleep(0.1)  # Simulate connection time

        self.servers[server_name] = server
        self.connected = True
        print(f"✓ Connected to {server_name}\n")

    async def list_server_capabilities(self, server_name: str):
        """Discover what a server can do."""
        if server_name not in self.servers:
            print(f"✗ Server {server_name} not connected")
            return

        server = self.servers[server_name]
        print(f"=== CAPABILITIES OF {server_name.upper()} ===\n")

        # List tools
        tools = await server.list_tools()
        print(f"Tools ({len(tools['tools'])}):")
        for tool in tools['tools']:
            print(f"  - {tool['name']}: {tool['description']}")
        print()

        # List resources
        resources = await server.list_resources()
        print(f"Resources ({len(resources['resources'])}):")
        for resource in resources['resources']:
            print(f"  - {resource['name']} ({resource['uri']})")
        print()

        # List prompts
        prompts = await server.list_prompts()
        print(f"Prompts ({len(prompts['prompts'])}):")
        for prompt in prompts['prompts']:
            print(f"  - {prompt['name']}: {prompt['description']}")
        print()

    async def get_tool_schema(self, server_name: str, tool_name: str):
        """Get the input schema for a specific tool."""
        if server_name not in self.servers:
            print(f"✗ Server {server_name} not connected")
            return None

        server = self.servers[server_name]
        tools = await server.list_tools()

        for tool in tools['tools']:
            if tool['name'] == tool_name:
                print(f"=== TOOL SCHEMA: {tool_name} ===\n")
                print(f"Description: {tool['description']}\n")
                print("Input Schema:")
                print(json.dumps(tool['inputSchema'], indent=2))
                return tool['inputSchema']

        print(f"✗ Tool {tool_name} not found")
        return None


def create_filesystem_server() -> MockMCPServer:
    """Create a mock filesystem MCP server."""
    server = MockMCPServer("filesystem")

    # Add tools
    server.add_tool(
        name="read_file",
        description="Read the contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }
    )

    server.add_tool(
        name="write_file",
        description="Write content to a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    )

    server.add_tool(
        name="list_directory",
        description="List contents of a directory",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory"
                }
            },
            "required": ["path"]
        }
    )

    # Add resources
    server.add_resource(
        uri="file:///home/user/documents",
        name="Documents Folder",
        mime_type="application/x-directory"
    )

    return server


def create_database_server() -> MockMCPServer:
    """Create a mock database MCP server."""
    server = MockMCPServer("database")

    # Add tools
    server.add_tool(
        name="query",
        description="Execute a SQL query",
        input_schema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute"
                },
                "params": {
                    "type": "array",
                    "description": "Query parameters",
                    "items": {"type": "string"}
                }
            },
            "required": ["sql"]
        }
    )

    server.add_tool(
        name="list_tables",
        description="List all tables in the database",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )

    server.add_tool(
        name="describe_table",
        description="Get schema information for a table",
        input_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table"
                }
            },
            "required": ["table_name"]
        }
    )

    # Add resources
    server.add_resource(
        uri="postgres://localhost/mydb/users",
        name="Users Table",
        mime_type="application/x-sql-table"
    )

    server.add_resource(
        uri="postgres://localhost/mydb/orders",
        name="Orders Table",
        mime_type="application/x-sql-table"
    )

    # Add prompts
    server.add_prompt(
        name="generate_query",
        description="Generate a SQL query from natural language",
        arguments=[
            {
                "name": "question",
                "description": "Natural language question",
                "required": True
            }
        ]
    )

    return server


def create_slack_server() -> MockMCPServer:
    """Create a mock Slack MCP server."""
    server = MockMCPServer("slack")

    # Add tools
    server.add_tool(
        name="send_message",
        description="Send a message to a Slack channel",
        input_schema={
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name or ID"
                },
                "text": {
                    "type": "string",
                    "description": "Message text"
                }
            },
            "required": ["channel", "text"]
        }
    )

    server.add_tool(
        name="search_messages",
        description="Search for messages in Slack",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    )

    # Add resources
    server.add_resource(
        uri="slack://channel/general",
        name="General Channel",
        mime_type="application/x-slack-channel"
    )

    return server


async def demo_single_server():
    """Demo: Connect to a single MCP server."""
    print("=" * 70)
    print("=== DEMO 1: CONNECTING TO A SINGLE SERVER ===\n")

    client = MCPClient()

    # Create and connect to filesystem server
    fs_server = create_filesystem_server()
    await client.connect_to_server("filesystem", fs_server)

    # Discover capabilities
    await client.list_server_capabilities("filesystem")

    # Get detailed schema for a specific tool
    await client.get_tool_schema("filesystem", "read_file")


async def demo_multiple_servers():
    """Demo: Connect to multiple MCP servers."""
    print("\n" + "=" * 70)
    print("=== DEMO 2: CONNECTING TO MULTIPLE SERVERS ===\n")

    client = MCPClient()

    # Create servers
    fs_server = create_filesystem_server()
    db_server = create_database_server()
    slack_server = create_slack_server()

    # Connect to all servers
    await client.connect_to_server("filesystem", fs_server)
    await client.connect_to_server("database", db_server)
    await client.connect_to_server("slack", slack_server)

    # Show capabilities of each
    for server_name in client.servers:
        await client.list_server_capabilities(server_name)


async def demo_tool_discovery():
    """Demo: Discover and inspect tools across servers."""
    print("=" * 70)
    print("=== DEMO 3: TOOL DISCOVERY ===\n")

    client = MCPClient()

    # Connect to servers
    db_server = create_database_server()
    await client.connect_to_server("database", db_server)

    # List all tools with detailed schemas
    print("Inspecting all database tools:\n")

    tools = await db_server.list_tools()
    for tool in tools['tools']:
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print("Schema:")
        print(json.dumps(tool['inputSchema'], indent=2))
        print("-" * 70)
        print()


def show_production_example():
    """Show how to use the real MCP library."""
    print("\n" + "=" * 70)
    print("=== PRODUCTION EXAMPLE ===\n")

    print("With the real mcp library, you would:\n")

    example_code = '''
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

# Configure server
server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/allowed/directory"
    ]
)

# Connect to server
async with stdio_client(server_params) as (read, write):
    async with Client(read, write) as session:
        # Initialize
        await session.initialize()

        # List available tools
        tools = await session.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        # Call a tool
        result = await session.call_tool(
            "read_file",
            {"path": "/path/to/file.txt"}
        )
        print(f"File contents: {result.content}")
'''

    print(example_code)

    print("\nKey differences from our mock:")
    print("  1. Real servers run as separate processes")
    print("  2. Communication via stdio (stdin/stdout)")
    print("  3. Full JSON-RPC 2.0 protocol implementation")
    print("  4. Proper error handling and validation")


def show_server_configuration():
    """Show how servers are typically configured."""
    print("\n" + "=" * 70)
    print("=== SERVER CONFIGURATION ===\n")

    print("MCP servers are typically configured in your app config:")
    print()

    config_example = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Documents"
                ]
            },
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-github"
                ],
                "env": {
                    "GITHUB_TOKEN": "your_token_here"
                }
            },
            "postgres": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-postgres",
                    "postgresql://localhost/mydb"
                ]
            }
        }
    }

    print(json.dumps(config_example, indent=2))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 4.6: MCP CLIENT")
    print("=" * 70 + "\n")

    # Run async demos
    asyncio.run(demo_single_server())
    asyncio.run(demo_multiple_servers())
    asyncio.run(demo_tool_discovery())

    # Show production examples
    show_production_example()
    show_server_configuration()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("""
1. MCP clients connect to servers via stdio transport
2. Servers expose tools, resources, and prompts
3. Discovery is dynamic - query servers for capabilities
4. Multiple servers can be connected simultaneously
5. Each server provides a focused set of capabilities

Next: 03_mcp_tool_use.py - Learn how to call tools via MCP
    """)
    print("=" * 70)

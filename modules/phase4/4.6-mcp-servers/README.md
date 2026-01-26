# Module 4.6: MCP Servers

> *"Standardize how AI agents access tools and data"*

This module covers the Model Context Protocol (MCP) - an open protocol that enables LLMs to securely connect to data sources and tools through a standardized client-server architecture.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_mcp_overview.py` | MCP Overview | Understanding MCP architecture and benefits |
| `02_mcp_client.py` | MCP Client | Connect to servers and discover capabilities |
| `03_mcp_tool_use.py` | Using MCP Tools | Call tools and build workflows |
| `04_custom_mcp_server.py` | Custom MCP Server | Build and deploy custom servers |

## What is MCP?

Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. Think of it as **USB-C for AI agents** - one standard protocol that connects to many different tools and data sources.

### Why MCP?

**Traditional Approach:**
```
AI App → Custom Code → File System
AI App → Custom Code → Database
AI App → Custom Code → Slack API
AI App → Custom Code → GitHub API
```
- Every integration requires custom code
- Hard to maintain and update
- Difficult to share across applications

**MCP Approach:**
```
AI App → MCP Client → MCP Server (FS)
                   → MCP Server (DB)
                   → MCP Server (Slack)
                   → MCP Server (GitHub)
```
- One client connects to all servers
- Standard protocol for all integrations
- Servers are reusable across applications

## Core Concepts

### 1. MCP Architecture

```
┌─────────────────────────────────────────────┐
│           YOUR AI APPLICATION                │
│        (Claude, GPT, Custom Agent)           │
└────────────────────┬────────────────────────┘
                     │
            ┌────────▼────────┐
            │   MCP CLIENT    │  ← Your app uses this
            │  (mcp library)  │
            └────────┬────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
 ┌────▼───┐    ┌────▼───┐    ┌────▼───┐
 │ MCP    │    │ MCP    │    │ MCP    │
 │ Server │    │ Server │    │ Server │
 │  (FS)  │    │  (DB)  │    │  (API) │
 └────┬───┘    └────┬───┘    └────┬───┘
      │             │             │
 ┌────▼───┐    ┌────▼───┐    ┌────▼───┐
 │  File  │    │Postgres│    │ Slack  │
 │ System │    │Database│    │  API   │
 └────────┘    └────────┘    └────────┘
```

### 2. Server Capabilities

MCP servers can provide three types of capabilities:

**Resources:** Read-only data sources
- File contents
- Database records
- API responses
- Documentation

**Tools:** Actions the LLM can invoke
- Create/update files
- Execute queries
- Send messages
- Run calculations

**Prompts:** Reusable templates
- Code review format
- Bug report structure
- Analysis patterns
- Query templates

### 3. Protocol Basics

MCP uses JSON-RPC 2.0 for communication:

```python
# Client requests available tools
{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
}

# Server responds with tool definitions
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }
        ]
    }
}

# Client calls the tool
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "read_file",
        "arguments": {"path": "/data/file.txt"}
    },
    "id": 2
}
```

## Use Cases

### 1. Code Assistant
Connect to:
- Filesystem (read/write code)
- Git (commit, branch, diff)
- GitHub (create PR, issues)
- Documentation (search docs)

**Benefit:** One assistant with access to entire dev workflow

### 2. Data Analysis Agent
Connect to:
- PostgreSQL (query data)
- Filesystem (read CSV/Excel)
- Slack (notify team)
- Visualization (create charts)

**Benefit:** Unified access to all data sources

### 3. Customer Support Bot
Connect to:
- Knowledge base (search docs)
- Ticketing system (create/update)
- CRM (customer history)
- Email (send responses)

**Benefit:** Complete customer context in one place

## MCP Ecosystem

### Official Servers (by Anthropic)
- `@modelcontextprotocol/server-filesystem` - File system access
- `@modelcontextprotocol/server-github` - GitHub integration
- `@modelcontextprotocol/server-gitlab` - GitLab integration
- `@modelcontextprotocol/server-postgres` - PostgreSQL database
- `@modelcontextprotocol/server-sqlite` - SQLite database
- `@modelcontextprotocol/server-slack` - Slack integration
- `@modelcontextprotocol/server-google-drive` - Google Drive

### Community Servers
- Notion, Airtable, Jira, Confluence
- AWS, GCP, Azure integrations
- Email providers (Gmail, Outlook)
- Analytics platforms

### Custom Servers
- Your company's internal APIs
- Proprietary databases
- Legacy systems
- Business-specific tools

## Installation

### Install MCP Client Library
```bash
pip install mcp
```

### Install Official Servers
```bash
# Filesystem server
npm install -g @modelcontextprotocol/server-filesystem

# GitHub server
npm install -g @modelcontextprotocol/server-github

# PostgreSQL server
npm install -g @modelcontextprotocol/server-postgres
```

## Running the Examples

### 1. MCP Overview
```bash
# Learn about MCP architecture and concepts
python 01_mcp_overview.py
```

Covers:
- What is MCP and why it matters
- MCP vs traditional integration
- Architecture patterns
- Ecosystem overview

### 2. MCP Client
```bash
# Connect to MCP servers
python 02_mcp_client.py
```

Demonstrates:
- Connecting to servers
- Discovering capabilities
- Inspecting tool schemas
- Multiple server connections

### 3. MCP Tool Use
```bash
# Use tools via MCP
python 03_mcp_tool_use.py
```

Shows:
- Basic tool calling
- Multi-step workflows
- Cross-server workflows
- Error handling
- LLM integration patterns

### 4. Custom MCP Server
```bash
# Build your own MCP server
python 04_custom_mcp_server.py
```

Includes:
- Calculator server example
- Weather API server example
- Task management server example
- Production implementation guide
- Deployment instructions

## Integration Patterns

### Pattern 1: Basic Tool Use

```python
from mcp.client import Client
from mcp.client.stdio import stdio_client, StdioServerParameters

# Configure server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
)

# Connect and use
async with stdio_client(server_params) as (read, write):
    async with Client(read, write) as session:
        await session.initialize()

        # List tools
        tools = await session.list_tools()

        # Call tool
        result = await session.call_tool(
            "read_file",
            {"path": "/allowed/path/file.txt"}
        )
```

### Pattern 2: LLM Integration (Anthropic)

```python
from anthropic import Anthropic

client = Anthropic()

# Convert MCP tools to Anthropic format
tools = [
    {
        "name": "read_file",
        "description": "Read a file",
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

        # Continue conversation with result
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
```

### Pattern 3: Multiple Servers

```python
# Connect to multiple servers
servers = {
    "filesystem": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    ),
    "database": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres", "postgres://localhost/db"]
    ),
    "slack": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        env={"SLACK_TOKEN": os.getenv("SLACK_TOKEN")}
    )
}

# Use all servers together
async def workflow():
    # Query database
    data = await db_session.call_tool("query", {"sql": "SELECT * FROM users"})

    # Save to file
    await fs_session.call_tool("write_file", {
        "path": "/reports/users.json",
        "content": data
    })

    # Notify team
    await slack_session.call_tool("send_message", {
        "channel": "reports",
        "text": "User report generated"
    })
```

## Building Custom Servers

### Basic Structure

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create server
server = Server("my-server")

# Register tools
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Does something useful",
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
        return [TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")

# Run server
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Server Configuration

Add to your application's config:

```json
{
    "mcpServers": {
        "my-custom-server": {
            "command": "python",
            "args": ["-m", "my_custom_server"],
            "env": {
                "API_KEY": "secret_key"
            }
        }
    }
}
```

## Best Practices

### Tool Design
- Use descriptive names (verb_noun pattern)
- Provide detailed descriptions
- Define strict input schemas
- Return structured JSON data
- Keep tools focused and single-purpose

### Error Handling
- Validate all inputs
- Return clear error messages
- Use appropriate error types
- Log errors for debugging
- Handle edge cases gracefully

### Security
- Validate and sanitize inputs
- Implement authentication
- Use environment variables for secrets
- Audit all tool calls
- Limit resource access

### Performance
- Keep operations fast (<1 second)
- Implement caching
- Use async operations
- Set timeouts
- Monitor resource usage

### Testing
- Unit test each tool
- Test error cases
- Validate schemas
- Integration tests
- Load testing for production

## Common Patterns

### 1. Data Pipeline
```
Query DB → Transform → Save to File → Notify Team
```

### 2. Code Review
```
Read PR → Analyze Code → Check Tests → Post Review
```

### 3. Customer Query
```
Search KB → Query CRM → Draft Response → Send Email
```

### 4. Report Generation
```
Gather Data → Create Charts → Format Report → Distribute
```

## Benefits

### For Developers
- Write once, use everywhere
- Standard protocol reduces complexity
- Focus on business logic
- Easy to test and maintain

### For Organizations
- Secure, controlled access
- Audit trail of operations
- Central management
- Reduced integration costs

### For Users
- More capable AI assistants
- Access to your data
- Consistent experience
- Privacy and security built-in

## Troubleshooting

### Connection Issues
- Verify server is installed correctly
- Check command and args in config
- Ensure server has required permissions
- Review environment variables

### Tool Call Failures
- Validate arguments against schema
- Check server logs for errors
- Verify resource access permissions
- Test server independently

### Performance Issues
- Implement caching
- Optimize tool handlers
- Use async operations
- Monitor resource usage

## Advanced Topics

### Resource Management
```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="file:///data/report.pdf",
            name="Monthly Report",
            mimeType="application/pdf"
        )
    ]
```

### Prompt Templates
```python
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="code_review",
            description="Review code changes",
            arguments=[
                PromptArgument(
                    name="diff",
                    description="Git diff to review",
                    required=True
                )
            ]
        )
    ]
```

### Progress Notifications
```python
# Send progress updates during long operations
async def long_operation():
    await session.send_progress_notification(
        progress_token="op1",
        progress=50,
        total=100
    )
```

## Security Considerations

### Access Control
- Implement authentication where needed
- Validate all requests
- Use scoped permissions
- Audit access logs

### Data Protection
- Sanitize inputs and outputs
- Never log sensitive data
- Encrypt data in transit
- Use secure connections

### Rate Limiting
- Implement per-client limits
- Throttle expensive operations
- Monitor usage patterns
- Alert on anomalies

## Book References

- `AI_eng.6` - LLM application architecture and tooling

## Next Steps

After mastering MCP servers:
- **Module 4.7**: Cloud Deployment - Deploy MCP-enabled applications
- **Module 4.5**: Async & Background Jobs - Run MCP tools asynchronously
- **Module 5.1**: Fine-tuning - Use MCP to manage training data
- **Module 3.7**: Agent Systems - Build agents with MCP tools

## Resources

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Official Servers](https://github.com/modelcontextprotocol/servers)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)

## Key Takeaways

MCP is transformative for AI applications:

1. **Standardization**: One protocol for all integrations
2. **Reusability**: Write servers once, use everywhere
3. **Composability**: Mix and match servers as needed
4. **Security**: Built-in access control and auditing
5. **Simplicity**: Less code, easier maintenance

MCP enables AI agents to access the tools and data they need while maintaining security and control. It's the foundation for building powerful, flexible AI systems.

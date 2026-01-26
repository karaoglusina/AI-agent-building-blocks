"""
01 - MCP Overview
==================
Understanding the Model Context Protocol (MCP) architecture.

Key concept: MCP is an open protocol that standardizes how applications provide context to LLMs. It enables LLMs to securely connect to data sources and tools through a client-server architecture.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from typing import Dict, List, Any
import json


def explain_mcp_concept():
    """
    Explain what MCP is and why it matters.

    MCP (Model Context Protocol) is like a "universal adapter" for AI agents.
    Instead of building custom integrations for every tool and data source,
    MCP provides a standard protocol for LLMs to access external resources.
    """
    print("=" * 70)
    print("=== WHAT IS MCP? ===\n")

    print("Model Context Protocol (MCP) is an open protocol that:")
    print("  1. Standardizes how LLMs connect to external data sources")
    print("  2. Provides secure, controlled access to tools and resources")
    print("  3. Enables composable, reusable server implementations")
    print("  4. Separates concerns between AI applications and integrations")

    print("\nThink of MCP like USB-C:")
    print("  - ONE standard protocol")
    print("  - MANY compatible servers (like USB-C devices)")
    print("  - Applications connect to any MCP server without custom code")


def mcp_architecture():
    """Explain the MCP architecture pattern."""
    print("\n" + "=" * 70)
    print("=== MCP ARCHITECTURE ===\n")

    architecture = """
    ┌─────────────────────────────────────────────────────────┐
    │                    YOUR AI APPLICATION                   │
    │                   (Claude, GPT, Custom)                  │
    └────────────────────────┬────────────────────────────────┘
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
    """
    print(architecture)

    print("\nKey Components:")
    print("  1. MCP Client: Built into your AI application")
    print("  2. MCP Server: Provides access to specific resources")
    print("  3. Resources: Files, databases, APIs, tools, etc.")


def mcp_vs_traditional():
    """Compare MCP approach to traditional integration."""
    print("\n" + "=" * 70)
    print("=== MCP vs TRADITIONAL INTEGRATION ===\n")

    print("TRADITIONAL APPROACH:")
    print("  AI App → Custom Code → File System")
    print("  AI App → Custom Code → Database")
    print("  AI App → Custom Code → Slack API")
    print("  AI App → Custom Code → GitHub API")
    print("  ❌ Every integration requires custom code")
    print("  ❌ Hard to maintain and update")
    print("  ❌ Difficult to share across applications")

    print("\nMCP APPROACH:")
    print("  AI App → MCP Client → MCP Server (FS)")
    print("  AI App → MCP Client → MCP Server (DB)")
    print("  AI App → MCP Client → MCP Server (Slack)")
    print("  AI App → MCP Client → MCP Server (GitHub)")
    print("  ✓ One client connects to all servers")
    print("  ✓ Standard protocol for all integrations")
    print("  ✓ Servers are reusable across applications")


def mcp_capabilities():
    """Explain what MCP servers can provide."""
    print("\n" + "=" * 70)
    print("=== MCP SERVER CAPABILITIES ===\n")

    capabilities = {
        "Resources": {
            "description": "Read-only data sources",
            "examples": [
                "File contents",
                "Database records",
                "API responses",
                "Documentation"
            ]
        },
        "Tools": {
            "description": "Actions the LLM can invoke",
            "examples": [
                "Create/update files",
                "Execute database queries",
                "Send messages",
                "Run calculations"
            ]
        },
        "Prompts": {
            "description": "Reusable prompt templates",
            "examples": [
                "Code review template",
                "Bug report format",
                "Analysis structure",
                "Query patterns"
            ]
        }
    }

    for capability, details in capabilities.items():
        print(f"{capability}:")
        print(f"  {details['description']}")
        print("  Examples:")
        for example in details['examples']:
            print(f"    - {example}")
        print()


def mcp_protocol_basics():
    """Explain the core protocol concepts."""
    print("=" * 70)
    print("=== MCP PROTOCOL BASICS ===\n")

    print("MCP uses JSON-RPC 2.0 for client-server communication:")
    print()

    # Example: List tools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }

    list_tools_response = {
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
                            "path": {
                                "type": "string",
                                "description": "Path to file"
                            }
                        },
                        "required": ["path"]
                    }
                }
            ]
        }
    }

    print("1. Client Request (list available tools):")
    print(json.dumps(list_tools_request, indent=2))

    print("\n2. Server Response:")
    print(json.dumps(list_tools_response, indent=2))

    print("\n3. Client calls tool:")
    call_tool_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {
                "path": "/data/document.txt"
            }
        },
        "id": 2
    }
    print(json.dumps(call_tool_request, indent=2))


def mcp_use_cases():
    """Show practical use cases for MCP."""
    print("\n" + "=" * 70)
    print("=== MCP USE CASES ===\n")

    use_cases = [
        {
            "scenario": "Code Assistant",
            "servers": [
                "filesystem (read/write code)",
                "git (commit, branch, diff)",
                "github (create PR, issues)",
                "docs (search documentation)"
            ],
            "benefit": "One assistant, multiple tools"
        },
        {
            "scenario": "Data Analysis Agent",
            "servers": [
                "postgresql (query data)",
                "filesystem (read CSV/Excel)",
                "slack (notify team)",
                "visualization (create charts)"
            ],
            "benefit": "Access all data sources"
        },
        {
            "scenario": "Customer Support Bot",
            "servers": [
                "knowledge-base (search docs)",
                "ticketing (create/update tickets)",
                "crm (customer history)",
                "email (send responses)"
            ],
            "benefit": "Unified customer context"
        }
    ]

    for i, case in enumerate(use_cases, 1):
        print(f"{i}. {case['scenario']}")
        print("   MCP Servers needed:")
        for server in case['servers']:
            print(f"     - {server}")
        print(f"   Benefit: {case['benefit']}")
        print()


def mcp_ecosystem():
    """Show the MCP ecosystem."""
    print("=" * 70)
    print("=== MCP ECOSYSTEM ===\n")

    print("OFFICIAL MCP SERVERS (by Anthropic):")
    print("  - @modelcontextprotocol/server-filesystem")
    print("  - @modelcontextprotocol/server-github")
    print("  - @modelcontextprotocol/server-gitlab")
    print("  - @modelcontextprotocol/server-google-drive")
    print("  - @modelcontextprotocol/server-slack")
    print("  - @modelcontextprotocol/server-postgres")
    print("  - @modelcontextprotocol/server-sqlite")

    print("\nCOMMUNITY SERVERS:")
    print("  - Notion, Airtable, Jira, Confluence")
    print("  - AWS, GCP, Azure integrations")
    print("  - Email providers (Gmail, Outlook)")
    print("  - Analytics platforms")

    print("\nCUSTOM SERVERS:")
    print("  - Your company's internal APIs")
    print("  - Proprietary databases")
    print("  - Legacy systems")
    print("  - Business-specific tools")


def mcp_benefits():
    """Explain the benefits of using MCP."""
    print("\n" + "=" * 70)
    print("=== WHY USE MCP? ===\n")

    benefits = {
        "For Developers": [
            "Write integration once, use everywhere",
            "Standard protocol reduces complexity",
            "Focus on business logic, not plumbing",
            "Easy to test and debug"
        ],
        "For Organizations": [
            "Secure, controlled access to resources",
            "Audit trail of all tool usage",
            "Central management of integrations",
            "Reduce integration maintenance costs"
        ],
        "For Users": [
            "AI agents that can access your data",
            "Consistent experience across tools",
            "More capable assistants",
            "Privacy and security built-in"
        ]
    }

    for audience, points in benefits.items():
        print(f"{audience}:")
        for point in points:
            print(f"  ✓ {point}")
        print()


def mcp_getting_started():
    """Show how to get started with MCP."""
    print("=" * 70)
    print("=== GETTING STARTED WITH MCP ===\n")

    print("1. INSTALL MCP CLIENT:")
    print("   pip install mcp")
    print()

    print("2. CHOOSE OR BUILD SERVERS:")
    print("   Option A: Use existing servers (npm install @modelcontextprotocol/server-*)")
    print("   Option B: Build your own (see 04_custom_mcp_server.py)")
    print()

    print("3. CONNECT TO SERVERS:")
    print("   See 02_mcp_client.py for examples")
    print()

    print("4. CALL TOOLS:")
    print("   See 03_mcp_tool_use.py for examples")
    print()

    print("Next steps:")
    print("  → 02_mcp_client.py - Connect to an MCP server")
    print("  → 03_mcp_tool_use.py - Use tools via MCP")
    print("  → 04_custom_mcp_server.py - Build your own server")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 4.6: MODEL CONTEXT PROTOCOL (MCP)")
    print("=" * 70 + "\n")

    explain_mcp_concept()
    mcp_architecture()
    mcp_vs_traditional()
    mcp_capabilities()
    mcp_protocol_basics()
    mcp_use_cases()
    mcp_ecosystem()
    mcp_benefits()
    mcp_getting_started()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("=" * 70)
    print("""
MCP is like USB-C for AI agents:
  - ONE standard protocol
  - MANY compatible servers
  - ANY application can connect
  - SECURE and controlled access

This standardization enables:
  ✓ Reusable integrations
  ✓ Composable AI systems
  ✓ Easier maintenance
  ✓ Broader capabilities
    """)
    print("=" * 70)

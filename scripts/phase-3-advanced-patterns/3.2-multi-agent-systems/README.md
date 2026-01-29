# Module 3.2: Multi-Agent Systems

*Coordinate multiple specialized agents*

## Overview

This module teaches how to design and coordinate systems with multiple AI agents working together. You'll learn patterns for agent handoff, specialization, coordination, preference detection, and inter-agent communication.

## Scripts

### 01 - Agent Handoff
**File:** `01_agent_handoff.py`

Transfer conversations between specialized agents based on user intent.

**Key concept:** Different agents handle different conversation stages or domains.

**Run:**
```bash
python 01_agent_handoff.py
```

### 02 - Specialized Agents
**File:** `02_specialized_agents.py`

Create domain-specific agents with unique capabilities and tools.

**Key concept:** Agents can have distinct capabilities, knowledge, and tools for their domain.

**Run:**
```bash
python 02_specialized_agents.py
```

### 03 - Coordinator Pattern
**File:** `03_coordinator_pattern.py`

Orchestrator agent that decomposes tasks and delegates to worker agents.

**Key concept:** Central coordinator decomposes tasks and routes to appropriate agents.

**Run:**
```bash
python 03_coordinator_pattern.py
```

### 04 - Preference Detection Agent
**File:** `04_preference_detector.py`

Specialized agent that extracts and maintains user preference memory.

**Key concept:** Specialized agent extracts and maintains user preference memory.

**Run:**
```bash
python 04_preference_detector.py
```

### 05 - Agent Communication
**File:** `05_agent_communication.py`

Multiple agents share information and collaborate through messages.

**Key concept:** Agents communicate and pass context to achieve complex goals.

**Run:**
```bash
python 05_agent_communication.py
```

## Key Concepts

1. **Agent Handoff**: Route users to appropriate specialized agents based on intent
2. **Specialization**: Create agents with focused expertise and dedicated tools
3. **Coordination**: Use orchestrator patterns to manage complex multi-agent workflows
4. **Preference Memory**: Maintain persistent understanding of user preferences
5. **Inter-Agent Communication**: Enable agents to share information and collaborate

## Dependencies

- `openai` - OpenAI API client
- `pydantic` - Data validation and structured outputs

## Book References

- **AI Engineering (AI_eng.6)**: Multi-agent systems and coordination patterns
- **AI Engineering (AI_eng.10.5)**: Orchestration and workflow management

## Application to Job Data

Multi-agent systems excel at complex job search scenarios:

- **Coordinator + Search + Analysis**: Break down complex queries into search and analysis subtasks
- **Preference Agent**: Learn and remember user job preferences over time
- **Specialized Routing**: Route to search agents, application helpers, or career advisors based on intent
- **Collaborative Processing**: Agents work together to find, analyze, and recommend jobs

## Design Patterns

### Agent Handoff Pattern
```python
intent = classify_intent(user_input)
agent = agents[intent]
response = agent.respond(user_input)
```

### Coordinator Pattern
```python
plan = coordinator.plan_tasks(user_query)
for subtask, agent_name in zip(plan.subtasks, plan.agents):
    result = workers[agent_name].execute(subtask)
final_answer = synthesize(results)
```

### Agent Communication Pattern
```python
message = agent_a.send_message(to="agent_b", content="task", data={...})
network.send(message)
response = agent_b.process_and_respond()
```

## Next Steps

After completing this module, proceed to:
- **Module 3.3**: Advanced Memory - Sophisticated memory management techniques
- **Module 3.4**: Advanced RAG - Enhanced retrieval-augmented generation patterns

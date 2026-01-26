# Implementation Templates

This document provides templates and patterns for implementing the remaining Phase 4 and Phase 5 modules.

## Completed Modules

### Phase 3: ✅ COMPLETE (50 scripts)
- 3.1: Advanced Agent Patterns ✅
- 3.2: Multi-Agent Systems ✅
- 3.3: Advanced Memory ✅
- 3.4: Advanced RAG ✅
- 3.5: Iterative Processing ✅
- 3.6: FastAPI Basics ✅
- 3.7: Clustering & Topics ✅
- 3.8: Evaluation Systems ✅
- 3.9: Document Processing ✅

### Phase 4: Docker & Containerization ✅
- 4.1: Docker & Containerization ✅ (5 files complete with README)

## Remaining Phase 4 Modules

### 4.2: PostgreSQL + pgvector (6 scripts + README)

**Scripts to create**:
1. `01_sqlalchemy_basics.py` - Define models with SQLAlchemy ORM
2. `02_crud_operations.py` - Create, Read, Update, Delete operations
3. `03_alembic_migrations.py` - Database schema versioning with Alembic
4. `04_pgvector_setup.py` - Enable and configure pgvector extension
5. `05_vector_search_pg.py` - Semantic search using pgvector
6. `06_hybrid_pg.py` - Combine full-text search + vector search

**Pattern**: Each script demonstrates PostgreSQL patterns for AI applications.

**Key concepts**:
- ORM models for job data with vector columns
- Database migrations for schema changes
- pgvector for production vector search
- Hybrid search combining keyword + semantic

### 4.3: Observability (5 scripts + README)

**Scripts to create**:
1. `01_langfuse_setup.py` - Initialize Langfuse client and configuration
2. `02_trace_llm_calls.py` - Trace all LLM API calls with metadata
3. `03_trace_rag_pipeline.py` - End-to-end RAG observability
4. `04_cost_monitoring.py` - Track token usage and costs in production
5. `05_custom_metrics.py` - Add application-specific metrics

**Pattern**: Langfuse integration for production monitoring.

**Key concepts**:
- Distributed tracing for LLM calls
- Cost tracking per user/session
- Performance metrics (latency, throughput)
- Debug production issues with traces

### 4.4: Guardrails (8 scripts + README)

**Scripts to create**:
1. `01_input_validation.py` - Validate user input before processing
2. `02_prompt_injection.py` - Detect and block prompt injection
3. `03_jailbreak_defense.py` - Protect against jailbreak attempts
4. `04_pii_filtering.py` - Detect and redact personal information
5. `05_output_validation.py` - Validate LLM outputs before returning
6. `06_content_moderation.py` - Filter inappropriate content
7. `07_guardrail_architecture.py` - Structure guardrails in system
8. `08_model_gateway.py` - Centralized LLM access with controls

**Pattern**: Security and safety layers for AI applications.

**Key concepts**:
- Input sanitization and validation
- Injection attack detection
- PII detection with regex + NER
- Output validation before user sees it
- Moderation API usage
- Gateway pattern for centralized control

### 4.5: Async & Background Jobs (5 scripts + README)

**Scripts to create**:
1. `01_asyncio_basics.py` - Async/await fundamentals for I/O
2. `02_concurrent_llm.py` - Multiple LLM calls in parallel
3. `03_celery_setup.py` - Configure Celery with Redis broker
4. `04_background_task.py` - Run long tasks asynchronously
5. `05_task_status.py` - Monitor background job progress

**Pattern**: Asynchronous processing for scalability.

**Key concepts**:
- asyncio for concurrent API calls
- Celery for heavy background processing
- Task queues for batch jobs
- Progress tracking and status updates
- Error handling in async context

### 4.6: MCP Servers (4 scripts + README)

**Scripts to create**:
1. `01_mcp_overview.py` - What MCP servers are and why use them
2. `02_mcp_client.py` - Connect to MCP server from application
3. `03_mcp_tool_use.py` - Call tools via MCP protocol
4. `04_custom_mcp_server.py` - Build custom MCP server

**Pattern**: Model Context Protocol for tool integration.

**Key concepts**:
- Standardized tool calling protocol
- Connect to external services via MCP
- Build MCP servers for custom tools
- Protocol specifications

### 4.7: Cloud Deployment (4 files + README)

**Files to create**:
1. `01_vm_setup.md` - Guide for setting up cloud VM
2. `02_https_caddy.md` - Configure HTTPS with Caddy reverse proxy
3. `03_health_checks.py` - Implement health check endpoints
4. `04_logging_config.py` - Production logging configuration

**Pattern**: Deploy to cloud (AWS, GCP, Digital Ocean).

**Key concepts**:
- VM provisioning and setup
- HTTPS with automatic certificates
- Load balancing considerations
- Logging and monitoring setup

### 4.8: CI/CD Basics (3 yml files + README)

**Files to create**:
1. `01_github_actions.yml` - Basic CI workflow
2. `02_automated_tests.yml` - Run tests on push/PR
3. `03_deploy_workflow.yml` - Deploy on merge to main

**Pattern**: Automated testing and deployment.

**Key concepts**:
- GitHub Actions workflows
- Automated testing pipelines
- Deployment automation
- Environment-specific configs

## Remaining Phase 5 Modules

### 5.1: Fine-tuning LLMs (6 scripts + README)

**Scripts to create**:
1. `01_when_to_finetune.py` - Decision framework: finetune vs RAG vs prompting
2. `02_data_preparation.py` - Format training data, quality checks
3. `03_lora_basics.py` - Parameter-efficient fine-tuning with LoRA
4. `04_quantization.py` - Run models locally with reduced precision
5. `05_sft_rlhf_dpo_overview.py` - Alignment techniques overview
6. `06_evaluation.py` - Measure fine-tuning improvements

**Pattern**: When and how to fine-tune models.

**Key concepts**:
- Decision tree for fine-tuning
- Data preparation and quality
- LoRA/QLoRA for efficiency
- Quantization for local deployment
- Alignment techniques (SFT, RLHF, DPO)

### 5.2: Custom Embeddings (4 scripts + README)

**Scripts to create**:
1. `01_sentence_transformers.py` - Use local embedding models
2. `02_domain_adaptation_tsdae.py` - Adapt embeddings to domain with TSDAE
3. `03_embedding_evaluation.py` - Measure embedding quality
4. `04_bias_in_embeddings.py` - Awareness of embedding biases

**Pattern**: Custom domain-specific embeddings.

**Key concepts**:
- sentence-transformers library
- Unsupervised domain adaptation
- Embedding evaluation metrics
- Bias detection and mitigation

### 5.3: Advanced NLP (3 scripts + README)

**Scripts to create**:
1. `01_dependency_parsing.py` - Extract grammatical structure
2. `02_relation_extraction.py` - Extract relationships between entities
3. `03_coreference.py` - Resolve pronouns to entities

**Pattern**: Advanced NLP techniques with spaCy.

**Key concepts**:
- Dependency trees for grammar
- Relation extraction patterns
- Coreference resolution
- Beyond simple NER

### 5.4: Multimodal (4 scripts + README)

**Scripts to create**:
1. `01_vision_basics.py` - Analyze images with GPT-4V
2. `02_clip_basics.py` - Text-image similarity with CLIP
3. `03_image_text_search.py` - CLIP-based multimodal search
4. `04_document_vision.py` - Extract info from document images

**Pattern**: Multimodal AI with vision models.

**Key concepts**:
- GPT-4V for image understanding
- CLIP for text-image matching
- Multimodal search systems
- Document understanding with vision

## Script Template

Every script should follow this structure:

```python
"""
[Number] - [Title]
==================
[One-line description]

Key concept: [Main takeaway in one sentence]

Book reference: [References]
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
# Other imports

# Implementation (40-150 lines)

def main_function():
    """Docstring explaining the function."""
    # Implementation
    pass

if __name__ == "__main__":
    print("=== [TITLE] ===\n")

    # Demonstration
    result = main_function()

    print(f"\n{result}")
    print("\nKey insight: [Main takeaway]")
```

## README Template

Every module README should include:

```markdown
# Module X.Y: [Module Name]

> *"[One-line quote about the module]"*

[Brief overview paragraph]

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| ... | ... | ... |

## Why [Topic]?

[Explain why this module matters]

## Core Concepts

### 1. [Concept Name]
[Explanation with code example]

## Prerequisites

[Installation instructions]

## Running the Scripts

[How to run each script]

## Key Insights

[Main takeaways from module]

## Book References

[List of book references]

## Next Steps

[What to learn after this module]
```

## Implementation Priority

### High Priority (Core Production Skills)
1. 4.2: PostgreSQL + pgvector (production vector DB)
2. 4.3: Observability (Langfuse monitoring)
3. 4.4: Guardrails (security and safety)
4. 4.5: Async & Background Jobs (scalability)

### Medium Priority (Advanced Techniques)
1. 5.1: Fine-tuning (when needed)
2. 5.2: Custom Embeddings (domain-specific)
3. 5.3: Advanced NLP (specialized tasks)

### Lower Priority (Specialized)
1. 4.6: MCP Servers (emerging standard)
2. 4.7: Cloud Deployment (infrastructure)
3. 4.8: CI/CD (DevOps)
4. 5.4: Multimodal (when needed)

## Quick Reference Commands

### For creating a new module:
```bash
# Create directory
mkdir -p modules/phase4/4.X-module-name

# Create scripts
touch modules/phase4/4.X-module-name/{01,02,03,04,05}_script.py
touch modules/phase4/4.X-module-name/README.md

# Follow templates above for content
```

### For testing a module:
```bash
# Navigate to module
cd modules/phase4/4.X-module-name

# Run scripts in order
python 01_script.py
python 02_script.py
# etc.
```

## Conclusion

All remaining modules follow the same patterns demonstrated in Phase 3 and the completed Phase 4 Module 4.1. Each module is fully specified in `meta/CURRICULUM_SPEC.md` with:
- Exact script counts
- Book references
- Key concepts
- Technology stack

The curriculum is systematic, comprehensive, and ready for full implementation following these templates.

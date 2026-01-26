

- [ ] **Module 4.1: Docker & Containerization**
    
    - [ ] `01_dockerfile_basics.py` - Create Dockerfile for Python AI app
    - [ ] `02_docker_compose.py` - Multi-container setup
    - [ ] `03_environment_config.py` - Handle env vars in containers
    - [ ] `Dockerfile` - Production-ready Dockerfile
    - [ ] `docker-compose.yml` - App + PostgreSQL + Redis
- [ ] **Module 4.2: PostgreSQL + pgvector** `sqlalchemy`, `alembic`, `psycopg2-binary`, `pgvector`
    
    - [ ] `01_sqlalchemy_basics.py` - Define models, create tables
    - [ ] `02_crud_operations.py` - Create, read, update, delete operations
    - [ ] `03_alembic_migrations.py` - Database schema versioning
    - [ ] `04_pgvector_setup.py` - Enable vector extension
    - [ ] `05_vector_search_pg.py` - Semantic search with pgvector
    - [ ] `06_hybrid_pg.py` - Combine text + vector search in PostgreSQL
- [ ] **Module 4.3: Observability** `langfuse`
    
    - [ ] `01_langfuse_setup.py` - Initialize Langfuse client
    - [ ] `02_trace_llm_calls.py` - Log all LLM interactions
    - [ ] `03_trace_rag_pipeline.py` - End-to-end RAG observability
    - [ ] `04_cost_monitoring.py` - Track token usage and costs
    - [ ] `05_custom_metrics.py` - Add application-specific metrics
- [ ] **Module 4.4: Guardrails** `openai` (moderation), `pydantic`, `re`
    
    - [ ] `01_input_validation.py` - Validate user input before processing
    - [ ] `02_prompt_injection.py` - Detect and block injection attempts
    - [ ] `03_jailbreak_defense.py` - Protect against jailbreak attempts
    - [ ] `04_pii_filtering.py` - Detect and redact personal information
    - [ ] `05_output_validation.py` - Validate LLM outputs before returning
    - [ ] `06_content_moderation.py` - Filter inappropriate content
    - [ ] `07_guardrail_architecture.py` - Structure guardrails in your system
    - [ ] `08_model_gateway.py` - Centralized LLM access with controls
- [ ] **Module 4.5: Async & Background Jobs** `asyncio`, `celery`, `redis`
    
    - [ ] `01_asyncio_basics.py` - Async/await fundamentals
    - [ ] `02_concurrent_llm.py` - Multiple LLM calls in parallel
    - [ ] `03_celery_setup.py` - Configure Celery with Redis
    - [ ] `04_background_task.py` - Run long tasks asynchronously
    - [ ] `05_task_status.py` - Monitor background job progress
- [ ] **Module 4.6: MCP Servers** `mcp`
    
    - [ ] `01_mcp_overview.py` - What MCP servers are and why
    - [ ] `02_mcp_client.py` - Connect to MCP server
    - [ ] `03_mcp_tool_use.py` - Call tools via MCP
    - [ ] `04_custom_mcp_server.py` - Build your own MCP server
- [ ] **Module 4.7: Cloud Deployment**
    
    - [ ] `01_vm_setup.md` - Set up cloud VM (documented)
    - [ ] `02_https_caddy.md` - Configure reverse proxy with HTTPS
    - [ ] `03_health_checks.py` - Implement health checks
    - [ ] `04_logging_config.py` - Production logging setup
- [ ] **Module 4.8: CI/CD Basics**
    
    - [ ] `01_github_actions.yml` - Simple CI workflow
    - [ ] `02_automated_tests.yml` - Run tests on push
    - [ ] `03_deploy_workflow.yml` - Deploy on merge to main
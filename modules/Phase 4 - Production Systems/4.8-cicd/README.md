# Module 4.8: CI/CD Basics

> *"Automate testing and deployment - ship faster with confidence"*

This module covers setting up Continuous Integration and Continuous Deployment (CI/CD) for AI applications using GitHub Actions.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_github_actions.yml` | GitHub Actions Basics | Understand workflow fundamentals |
| `02_automated_tests.yml` | Automated Testing | Run tests on every push |
| `03_deploy_workflow.yml` | Deploy Workflow | Deploy on merge to main |

## Why CI/CD?

**Benefits:**
- **Catch bugs early**: Tests run on every change
- **Consistent quality**: Automated checks ensure standards
- **Fast feedback**: Know within minutes if code works
- **Safe deployments**: Tests pass before production
- **Reduced errors**: Automation eliminates manual mistakes
- **Time savings**: No manual testing or deployment

**AI Application Specific:**
- Test with mocked LLM responses (fast, free)
- Validate prompt templates automatically
- Check for PII in logs
- Monitor token usage
- Test guardrails

## Core Concepts

### 1. CI/CD Pipeline

```
Developer Push
    │
    ├─→ Trigger CI Pipeline
    │   │
    │   ├─→ Checkout Code
    │   ├─→ Install Dependencies
    │   ├─→ Lint Code
    │   ├─→ Run Unit Tests
    │   ├─→ Run Integration Tests
    │   ├─→ Security Scan
    │   └─→ Generate Reports
    │
    ├─→ If tests pass on main branch
    │   │
    │   ├─→ Trigger CD Pipeline
    │   │   │
    │   │   ├─→ Build Application
    │   │   ├─→ Deploy to Production
    │   │   ├─→ Run Health Checks
    │   │   ├─→ Run Smoke Tests
    │   │   └─→ Notify Team
    │   │
    │   └─→ If deployment fails
    │       └─→ Rollback Automatically
    │
    └─→ If tests fail
        └─→ Block merge, notify developer
```

### 2. GitHub Actions Architecture

```
.github/workflows/
    │
    ├─→ test.yml          # Run tests on every push
    ├─→ deploy.yml        # Deploy on merge to main
    └─→ schedule.yml      # Scheduled tasks

Each workflow contains:
    │
    ├─→ Triggers (on:)
    │   ├─ push
    │   ├─ pull_request
    │   ├─ schedule
    │   └─ workflow_dispatch
    │
    ├─→ Jobs (run in parallel)
    │   ├─ lint
    │   ├─ test
    │   ├─ build
    │   └─ deploy
    │
    └─→ Steps (run sequentially)
        ├─ Checkout code
        ├─ Set up Python
        ├─ Install dependencies
        └─ Run commands
```

### 3. Testing Pyramid for AI Apps

```
               ┌─────────────┐
               │  E2E Tests  │ ← Few, real API calls
               │  (slow)     │
               └─────────────┘
              ┌───────────────┐
              │ Integration   │ ← Some, mocked APIs
              │   Tests       │
              └───────────────┘
           ┌────────────────────┐
           │   Unit Tests       │ ← Many, fast, isolated
           │   (mocked)         │
           └────────────────────┘
```

**For AI Applications:**

```python
# Unit tests (fast, no API calls)
def test_prompt_template():
    prompt = format_prompt("test", {"context": "data"})
    assert "test" in prompt
    assert "data" in prompt

# Integration tests (mocked API)
@patch('openai.ChatCompletion.create')
def test_llm_call(mock_create):
    mock_create.return_value = {"choices": [...]}
    result = call_llm("test")
    assert result is not None

# E2E tests (real API, only on main)
@pytest.mark.integration
def test_rag_pipeline():
    result = rag_query("What is RAG?")
    assert len(result.sources) > 0
```

## GitHub Actions Workflow Structure

### Basic Workflow Anatomy

```yaml
name: Workflow Name

# When to trigger
on:
  push:
    branches: [ main ]
  pull_request:

# Environment variables
env:
  PYTHON_VERSION: "3.11"

# Jobs (run in parallel by default)
jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install -r requirements.txt
      - run: pytest
```

### Common Triggers

```yaml
# Run on push to specific branches
on:
  push:
    branches: [ main, develop ]

# Run on pull requests
on:
  pull_request:
    branches: [ main ]

# Run on schedule (cron)
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

# Manual trigger
on:
  workflow_dispatch:

# Multiple triggers
on:
  push:
    branches: [ main ]
  pull_request:
  workflow_dispatch:
```

### Job Dependencies

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest

  deploy:
    runs-on: ubuntu-latest
    needs: test  # Wait for test job
    if: github.ref == 'refs/heads/main'
    steps:
      - run: ./deploy.sh
```

### Matrix Strategy (Test Multiple Versions)

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest
```

## Testing in CI

### 1. Unit Tests (Fast, Mocked)

```python
# tests/unit/test_embeddings.py
import pytest
from unittest.mock import Mock, patch

@patch('openai.Embedding.create')
def test_generate_embeddings(mock_create):
    """Test embedding generation with mocked API."""
    mock_create.return_value = Mock(
        data=[Mock(embedding=[0.1, 0.2, 0.3])]
    )

    result = generate_embeddings("test text")

    assert len(result) == 3
    assert result == [0.1, 0.2, 0.3]
    mock_create.assert_called_once()
```

### 2. Integration Tests (Database, Services)

```yaml
# .github/workflows/test.yml
jobs:
  integration-tests:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/integration/
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/testdb
```

### 3. Test Configuration (pytest.ini)

```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

# Test markers
markers = [
    "integration: integration tests (use real services)",
    "slow: slow tests",
    "mock: tests with mocked responses",
]

# Skip integration tests by default
addopts = "-v -m 'not integration'"
```

### 4. Running Specific Tests

```bash
# All tests
pytest

# Only unit tests
pytest tests/unit/

# Only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# With coverage
pytest --cov=. --cov-report=html
```

## Deployment Workflows

### 1. Simple Deployment (SSH)

```yaml
- name: Deploy via SSH
  env:
    SSH_PRIVATE_KEY: ${{ secrets.SSH_PRIVATE_KEY }}
    SERVER_HOST: ${{ secrets.SERVER_HOST }}
  run: |
    # Set up SSH
    mkdir -p ~/.ssh
    echo "$SSH_PRIVATE_KEY" > ~/.ssh/id_rsa
    chmod 600 ~/.ssh/id_rsa
    ssh-keyscan -H $SERVER_HOST >> ~/.ssh/known_hosts

    # Deploy
    ssh user@$SERVER_HOST << 'ENDSSH'
      cd /home/aiapp/apps/myapp
      git pull origin main
      source .venv/bin/activate
      pip install -r requirements.txt
      sudo supervisorctl restart aiapp
    ENDSSH
```

### 2. Docker Deployment

```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: username/app:latest

- name: Deploy container
  run: |
    ssh user@server << 'ENDSSH'
      docker pull username/app:latest
      docker stop myapp || true
      docker rm myapp || true
      docker run -d --name myapp -p 8000:8000 username/app:latest
    ENDSSH
```

### 3. Blue-Green Deployment

```yaml
- name: Deploy new version (port 8001)
  run: |
    ssh user@server << 'ENDSSH'
      # Start new version on different port
      cd /home/aiapp/apps/myapp-new
      git pull origin main
      source .venv/bin/activate
      pip install -r requirements.txt
      uvicorn main:app --port 8001 &

      # Health check
      sleep 5
      curl -f http://localhost:8001/health

      # Switch Caddy to new port
      sudo systemctl reload caddy

      # Stop old version
      sudo supervisorctl stop aiapp
    ENDSSH
```

## Secrets Management

### Adding Secrets to GitHub

1. Go to repository **Settings**
2. Navigate to **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add secrets:

```
SSH_PRIVATE_KEY     - Private SSH key for server
SERVER_HOST         - Production server hostname
SERVER_USER         - SSH username
OPENAI_API_KEY      - OpenAI API key
DATABASE_URL        - Production database URL
DOCKER_USERNAME     - Docker Hub username
DOCKER_PASSWORD     - Docker Hub token
```

### Using Secrets in Workflows

```yaml
- name: Use secret
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    # Secret is available as environment variable
    python script.py

- name: Deploy with secret
  run: |
    echo "${{ secrets.SSH_PRIVATE_KEY }}" > key.pem
    chmod 600 key.pem
    ssh -i key.pem user@server "deploy"
```

**Important:**
- Secrets are automatically masked in logs
- Never echo secrets directly
- Don't commit secrets to repository

## Common Patterns

### 1. Caching Dependencies

```yaml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### 2. Conditional Execution

```yaml
# Only on main branch
- name: Deploy
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh

# Only on pull requests
- name: Comment coverage
  if: github.event_name == 'pull_request'
  run: ./comment-coverage.sh

# Only on schedule
- name: Daily backup
  if: github.event_name == 'schedule'
  run: ./backup.sh
```

### 3. Artifacts (Share Between Jobs)

```yaml
# Job 1: Create artifact
- name: Build
  run: python setup.py build

- name: Upload artifact
  uses: actions/upload-artifact@v4
  with:
    name: build-output
    path: dist/

# Job 2: Use artifact
- name: Download artifact
  uses: actions/download-artifact@v4
  with:
    name: build-output
```

### 4. Status Badges

Add to README.md:

```markdown
![Tests](https://github.com/user/repo/actions/workflows/test.yml/badge.svg)
![Deploy](https://github.com/user/repo/actions/workflows/deploy.yml/badge.svg)
```

## AI-Specific CI/CD Considerations

### 1. Mocked LLM Tests

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_openai():
    """Mock OpenAI responses for testing."""
    with patch('openai.ChatCompletion.create') as mock:
        mock.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
        yield mock
```

### 2. Token Usage Tracking

```yaml
- name: Test with token tracking
  run: |
    pytest --log-level=INFO
    cat token_usage.log
    # Alert if usage exceeds threshold
```

### 3. Prompt Testing

```python
def test_prompt_no_pii():
    """Ensure prompts don't leak PII."""
    prompt = generate_prompt(user_data)
    assert not contains_email(prompt)
    assert not contains_phone(prompt)
```

### 4. Cost Control

```yaml
- name: Check API costs
  if: github.event_name == 'push'
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    # Only run expensive tests on main branch
    pytest -m "not expensive"
```

## Troubleshooting

### Workflow Not Triggering

```bash
# Check workflow file location
# Must be in .github/workflows/

# Check YAML syntax
yamllint .github/workflows/test.yml

# Check GitHub Actions tab for errors
```

### Tests Pass Locally, Fail in CI

```bash
# Check environment differences
# - Python version
# - Dependencies version
# - OS (Linux vs Mac)

# Run tests in Docker locally
docker run -it python:3.11 bash
pip install -r requirements.txt
pytest
```

### Secrets Not Working

```bash
# Verify secret exists in Settings → Secrets
# Check secret name matches exactly
# Secrets are case-sensitive
```

### Deployment Fails

```bash
# Check SSH key permissions
chmod 600 ~/.ssh/id_rsa

# Test SSH connection manually
ssh -i key.pem user@server

# Check server logs
ssh user@server "sudo journalctl -u aiapp -n 50"
```

## Best Practices

### 1. Test Before Deploy

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest

  deploy:
    needs: test  # Deploy only if tests pass
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh
```

### 2. Use Environments

```yaml
deploy-production:
  environment:
    name: production
    url: https://yourdomain.com
  steps:
    - run: ./deploy.sh
```

Benefits:
- Required reviewers
- Deployment protection rules
- Environment-specific secrets

### 3. Fail Fast

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
  fail-fast: true  # Stop all on first failure
```

### 4. Cache Everything

```yaml
# Cache pip
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

# Cache models
- uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: ${{ runner.os }}-models-${{ hashFiles('**/model_config.json') }}
```

### 5. Meaningful Names

```yaml
# Bad
name: CI

# Good
name: Test, Build, and Deploy

jobs:
  # Bad
  job1:

  # Good
  run-unit-tests:
```

## Example: Complete CI/CD Setup

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest --cov=.

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        env:
          SSH_KEY: ${{ secrets.SSH_PRIVATE_KEY }}
        run: |
          echo "$SSH_KEY" > key.pem
          chmod 600 key.pem
          ssh -i key.pem user@server 'cd app && git pull && supervisorctl restart app'
      - name: Health check
        run: curl -f https://yourdomain.com/health
```

## Book References

- `AI_eng.4` - Testing AI applications

## Next Steps

After mastering CI/CD:
- **Module 4.7**: Cloud Deployment - Deploy infrastructure
- **Module 4.3**: Observability - Monitor deployed apps
- **Module 5.1**: Fine-tuning - Deploy custom models
- Advanced: Kubernetes, ArgoCD for complex deployments

## Additional Resources

- GitHub Actions docs: https://docs.github.com/en/actions
- GitHub Actions Marketplace: https://github.com/marketplace?type=actions
- pytest docs: https://docs.pytest.org/
- Docker GitHub Actions: https://github.com/docker/build-push-action
- Secrets management: https://docs.github.com/en/actions/security-guides/encrypted-secrets

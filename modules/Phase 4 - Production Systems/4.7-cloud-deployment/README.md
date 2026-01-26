# Module 4.7: Cloud Deployment

> *"From localhost to production - deploy your AI applications to the cloud"*

This module covers deploying AI applications to production cloud environments - from VM setup and HTTPS configuration to health checks and logging.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_vm_setup.md` | VM Setup Guide | Configure cloud server for production |
| `02_https_caddy.md` | HTTPS with Caddy | Automatic HTTPS reverse proxy |
| `03_health_checks.py` | Health Check Endpoint | Monitor application health |
| `04_logging_config.py` | Production Logging | Structured logging for production |

## Why Cloud Deployment?

Moving from development to production requires:
- **Reliability**: High uptime and fault tolerance
- **Security**: HTTPS, firewalls, access control
- **Scalability**: Handle increasing traffic
- **Monitoring**: Know when things break
- **Automation**: Repeatable deployments

## Core Concepts

### 1. Deployment Architecture

```
Internet
    │
    ├─→ DNS (yourdomain.com)
    │
    └─→ Cloud VM (DigitalOcean/AWS/GCP)
         │
         ├─→ Firewall (UFW)
         │   ├─ Allow: 80 (HTTP)
         │   ├─ Allow: 443 (HTTPS)
         │   └─ Allow: 22 (SSH)
         │
         ├─→ Reverse Proxy (Caddy/Nginx)
         │   ├─ SSL/TLS termination
         │   ├─ Request routing
         │   └─ Load balancing
         │
         ├─→ Process Manager (Supervisor)
         │   ├─ Auto-restart on crash
         │   ├─ Log management
         │   └─ Process monitoring
         │
         └─→ AI Application (FastAPI)
             ├─ Health checks
             ├─ Structured logging
             └─ Error handling
```

### 2. The Deployment Stack

```
Layer 7: Application    → FastAPI + Python
Layer 6: Process Mgmt   → Supervisor
Layer 5: Reverse Proxy  → Caddy (HTTPS)
Layer 4: Firewall       → UFW
Layer 3: Operating Sys  → Ubuntu 22.04
Layer 2: Cloud VM       → DigitalOcean/AWS
Layer 1: Network        → DNS + Load Balancer
```

### 3. Request Flow

```
User Request (HTTPS)
    │
    ├─→ DNS Resolution
    │
    ├─→ SSL/TLS Handshake (Caddy)
    │
    ├─→ Reverse Proxy (Caddy)
    │   ├─ HTTPS → HTTP
    │   ├─ Add headers
    │   └─ Route request
    │
    ├─→ Application (FastAPI on :8000)
    │   ├─ Health check
    │   ├─ Process request
    │   └─ Generate response
    │
    └─→ Response (HTTPS to user)
```

## VM Setup Process

### 1. Choose Cloud Provider

**Beginner-friendly:**
- DigitalOcean: Simple UI, predictable pricing
- Linode: Developer-friendly
- Vultr: Good performance

**Enterprise:**
- AWS EC2: Most features, complex
- GCP Compute: AI/ML focused
- Azure: Windows integration

### 2. Server Specifications

```
Development/Testing:
- 2 vCPUs, 4GB RAM, 50GB SSD
- Cost: ~$20-40/month

Production (Small):
- 4 vCPUs, 8GB RAM, 100GB SSD
- Cost: ~$40-80/month

Production (Large):
- 8 vCPUs, 16GB RAM, 200GB SSD
- Cost: ~$100-200/month
```

### 3. Initial Configuration

```bash
# Update system
apt update && apt upgrade -y

# Create non-root user
adduser aiapp
usermod -aG sudo aiapp

# Install essentials
apt install -y python3.11 git curl nginx supervisor

# Configure firewall
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
```

### 4. Application Deployment

```bash
# Clone repository
git clone https://github.com/you/app.git
cd app

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cat > .env << EOF
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
EOF
```

## HTTPS with Caddy

### Why Caddy?

**Automatic HTTPS:**
- Gets SSL certificates automatically
- Renews certificates automatically
- Zero configuration needed

**Simple Configuration:**
```caddy
yourdomain.com {
    reverse_proxy localhost:8000
}
```

That's it! Caddy handles:
- SSL/TLS certificates (Let's Encrypt)
- HTTP → HTTPS redirect
- HTTP/2 and HTTP/3
- Certificate renewal

### Caddy vs Nginx

| Feature | Caddy | Nginx |
|---------|-------|-------|
| Auto HTTPS | Yes | No (needs Certbot) |
| Config syntax | Simple | Complex |
| Default HTTP/2 | Yes | Requires config |
| Certificate renewal | Automatic | Cron job |
| Learning curve | Easy | Moderate |

### Common Caddy Patterns

**Basic reverse proxy:**
```caddy
yourdomain.com {
    reverse_proxy localhost:8000
}
```

**With health checks:**
```caddy
yourdomain.com {
    reverse_proxy localhost:8000 {
        health_uri /health
        health_interval 30s
    }
}
```

**Multiple backends (load balancing):**
```caddy
yourdomain.com {
    reverse_proxy localhost:8000 localhost:8001 localhost:8002 {
        lb_policy round_robin
    }
}
```

**Long LLM requests:**
```caddy
yourdomain.com {
    reverse_proxy localhost:8000 {
        transport http {
            read_timeout 5m
            write_timeout 5m
        }
    }
}
```

## Health Checks

### Three Types of Health Checks

**1. Liveness Probe**
- Is the application alive?
- Returns 200 if process is running
- Kubernetes uses this to restart dead containers

```python
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}
```

**2. Readiness Probe**
- Is the application ready for traffic?
- Returns 200 only if dependencies are healthy
- Load balancers use this for routing

```python
@app.get("/health/ready")
async def readiness():
    if database.is_connected() and vector_db.is_healthy():
        return {"status": "ready"}
    raise HTTPException(503)
```

**3. Full Health Check**
- Detailed status of all components
- Used for monitoring and debugging

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "vector_db": "healthy",
            "cache": "degraded",
            "openai_api": "healthy"
        }
    }
```

### Health Check Best Practices

1. **Fast checks**: Complete in < 1 second
2. **No side effects**: Don't modify state
3. **Check dependencies**: Database, APIs, cache
4. **Return HTTP codes**: 200 (healthy), 503 (unhealthy)
5. **Include details**: What's broken and why

### Using Health Checks

**Load balancer configuration:**
```
Health check endpoint: /health/ready
Interval: 30 seconds
Timeout: 5 seconds
Unhealthy threshold: 2 consecutive failures
```

**Monitoring alerts:**
```
Alert: "Application unhealthy"
Condition: /health returns non-200 for 2+ minutes
Severity: Critical
```

## Production Logging

### Structured Logging (JSON)

**Bad (plain text):**
```
2024-01-26 10:30:45 - INFO - User logged in
```

**Good (structured JSON):**
```json
{
  "timestamp": "2024-01-26T10:30:45Z",
  "level": "INFO",
  "message": "User logged in",
  "user_id": "user_123",
  "ip_address": "203.0.113.42",
  "event_type": "auth"
}
```

**Why JSON?**
- Easy to parse and search
- Works with log aggregation tools
- Can query specific fields
- Machine-readable

### Log Types

**1. Application Logs**
```python
logger.info("Processing request", extra={
    "request_id": "req_123",
    "user_id": "user_456",
    "operation": "chat"
})
```

**2. Access Logs**
```python
logger.info("API request", extra={
    "method": "POST",
    "path": "/api/chat",
    "status_code": 200,
    "duration_ms": 1234
})
```

**3. Error Logs**
```python
logger.error("LLM call failed", extra={
    "model": "gpt-4",
    "error": str(e),
    "user_id": "user_123"
}, exc_info=True)
```

**4. Metrics Logs**
```python
logger.info("LLM call completed", extra={
    "model": "gpt-4o-mini",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "duration_ms": 1500,
    "cost_usd": 0.002
})
```

### Log Levels

```
DEBUG   → Detailed info for developers (not in production)
INFO    → General application flow
WARNING → Something unexpected but handled
ERROR   → Something failed but app continues
CRITICAL → Something failed and app may crash
```

### Log Rotation

Prevent logs from filling disk:

```python
# Rotate by size (100MB per file, keep 5 backups)
RotatingFileHandler(
    'app.log',
    maxBytes=100*1024*1024,
    backupCount=5
)

# Rotate by time (daily, keep 30 days)
TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    backupCount=30
)
```

### Searching Logs

**Using jq (JSON processor):**

```bash
# View logs with pretty printing
cat app.log | jq .

# Filter by level
cat app.log | jq 'select(.level=="ERROR")'

# Filter by user
cat app.log | jq 'select(.user_id=="user_123")'

# Filter by time range
cat app.log | jq 'select(.timestamp >= "2024-01-26T10:00:00Z")'

# Get LLM costs
cat app.log | jq 'select(.event_type=="llm_call") | .cost_usd' | awk '{sum+=$1} END {print sum}'
```

## Security Best Practices

### 1. Server Hardening

```bash
# Never use root for applications
adduser aiapp
su - aiapp

# Disable password authentication
# In /etc/ssh/sshd_config:
PasswordAuthentication no
PermitRootLogin no

# Enable firewall
ufw enable
ufw default deny incoming
ufw allow OpenSSH
ufw allow 'Nginx Full'
```

### 2. Secrets Management

**Never commit secrets:**
```bash
# Use environment variables
export OPENAI_API_KEY=sk-...

# Or .env file (not in git)
echo ".env" >> .gitignore
```

**Load from environment:**
```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

### 3. HTTPS Only

```caddy
# Caddy automatically redirects HTTP → HTTPS
yourdomain.com {
    reverse_proxy localhost:8000
}
```

### 4. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request):
    pass
```

## Monitoring and Alerts

### Key Metrics to Track

**Application Health:**
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (% of requests)
- Health check status

**LLM Usage:**
- Token usage per user
- Cost per request
- Average latency
- API errors

**Infrastructure:**
- CPU usage
- Memory usage
- Disk usage
- Network traffic

### Setting Up Alerts

**Examples:**
```
Alert: Error rate > 5%
Alert: Response time p95 > 5s
Alert: Disk usage > 80%
Alert: Health check fails for 2+ minutes
Alert: Daily LLM cost > $100
```

## Common Issues and Solutions

### Application Won't Start

```bash
# Check logs
sudo supervisorctl tail aiapp stderr

# Check port availability
sudo netstat -tlnp | grep 8000

# Check permissions
ls -la /home/aiapp/app
```

### Can't Access via Domain

```bash
# Check DNS
dig yourdomain.com

# Check firewall
sudo ufw status

# Check Caddy
sudo systemctl status caddy
sudo journalctl -u caddy -n 50
```

### High Memory Usage

```bash
# Check memory
free -h

# Top memory consumers
ps aux --sort=-%mem | head

# Add swap or upgrade instance
```

### Slow Responses

```bash
# Check CPU
htop

# Check disk I/O
iotop

# Check logs for slow queries
cat app.log | jq 'select(.duration_ms > 5000)'
```

## Cost Optimization

### 1. Right-size Your VM

- Start small, scale up as needed
- Monitor resource usage
- Don't overprovision

### 2. Use Reserved Instances

- 30-50% discount for 1-year commitment
- Good for stable workloads

### 3. Optimize LLM Usage

- Cache responses when possible
- Use cheaper models (gpt-4o-mini vs gpt-4)
- Implement rate limiting
- Set budget alerts

### 4. Log Rotation

- Don't let logs fill disk
- Compress old logs
- Delete after retention period

## Deployment Checklist

Before going to production:

- [ ] Domain configured and DNS propagated
- [ ] HTTPS working (green padlock)
- [ ] Health checks returning 200
- [ ] Logs being written to files
- [ ] Log rotation configured
- [ ] Firewall enabled and configured
- [ ] Application running under non-root user
- [ ] Secrets in environment variables
- [ ] Process manager (Supervisor) configured
- [ ] Monitoring and alerts set up
- [ ] Backups configured (if stateful)
- [ ] SSH password auth disabled
- [ ] Rate limiting enabled
- [ ] Error tracking set up
- [ ] Cost alerts configured

## Running the Examples

### 1. VM Setup

Follow the guide in `01_vm_setup.md`:
```bash
# SSH into server
ssh user@your-server-ip

# Run setup commands
```

### 2. Install Caddy

Follow `02_https_caddy.md`:
```bash
# Install Caddy
sudo apt install caddy

# Configure
sudo nano /etc/caddy/Caddyfile

# Restart
sudo systemctl restart caddy
```

### 3. Test Health Checks

```bash
# Run the health check server
uvicorn modules.phase4.4.7-cloud-deployment.03_health_checks:app --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

### 4. Test Logging

```bash
# Run logging demo
python modules/phase4/4.7-cloud-deployment/04_logging_config.py

# View logs
cat /tmp/aiapp_logs/demo_app.log | jq .
```

## Book References

- `AI_eng.10` - Production deployment and operations

## Next Steps

After mastering cloud deployment:
- **Module 4.8**: CI/CD - Automate deployments
- **Module 4.3**: Observability - Monitor with Langfuse
- **Module 4.5**: Async & Background Jobs - Scale with workers
- **Module 5.1**: Fine-tuning - Deploy custom models

## Additional Resources

- DigitalOcean Tutorials: https://www.digitalocean.com/community/tutorials
- Caddy Documentation: https://caddyserver.com/docs/
- Let's Encrypt: https://letsencrypt.org/
- Supervisor Documentation: http://supervisord.org/
- Ubuntu Server Guide: https://ubuntu.com/server/docs
- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/

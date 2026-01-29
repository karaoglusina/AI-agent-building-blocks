# VM Setup Guide for AI Applications

This guide walks through setting up a cloud VM for deploying AI applications in production.

## Prerequisites

- Cloud provider account (AWS, GCP, Azure, DigitalOcean, etc.)
- SSH key pair for secure access
- Domain name (optional, but recommended)

## Choosing a Cloud Provider

### Popular Options

**DigitalOcean** (Beginner-friendly)
- Simple interface
- Predictable pricing
- Good documentation
- Droplets from $6/month

**AWS EC2** (Enterprise-grade)
- Most features
- Complex pricing
- Global infrastructure
- Free tier available

**GCP Compute Engine** (AI/ML focused)
- Good GPU support
- Competitive pricing
- Integration with GCP AI services
- Free tier available

**Azure Virtual Machines** (Enterprise)
- Windows integration
- OpenAI service integration
- Enterprise features
- Free tier available

## VM Specifications for AI Apps

### Small Application (Development/Testing)
```
CPU: 2 vCPUs
RAM: 4 GB
Storage: 50 GB SSD
Cost: ~$20-40/month
Good for: Testing, small-scale deployments
```

### Medium Application (Production)
```
CPU: 4 vCPUs
RAM: 8 GB
Storage: 100 GB SSD
Cost: ~$40-80/month
Good for: Production apps with moderate traffic
```

### Large Application (High Traffic)
```
CPU: 8 vCPUs
RAM: 16 GB
Storage: 200 GB SSD
Cost: ~$100-200/month
Good for: High traffic, multiple services
```

## Step-by-Step Setup (DigitalOcean Example)

### 1. Create Droplet

```bash
# Via Web UI:
# - Click "Create" → "Droplets"
# - Choose Ubuntu 22.04 LTS
# - Select plan (e.g., 4GB RAM / 2 vCPUs)
# - Choose datacenter region (closest to users)
# - Add SSH key
# - Create Droplet

# Via CLI (doctl):
doctl compute droplet create ai-app-prod \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc1 \
  --ssh-keys YOUR_KEY_ID
```

### 2. Initial Server Configuration

```bash
# SSH into server
ssh root@YOUR_DROPLET_IP

# Update system packages
apt update && apt upgrade -y

# Set timezone
timedatectl set-timezone America/New_York

# Create non-root user
adduser aiapp
usermod -aG sudo aiapp

# Copy SSH keys to new user
rsync --archive --chown=aiapp:aiapp ~/.ssh /home/aiapp
```

### 3. Install Essential Software

```bash
# Switch to app user
su - aiapp

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install system dependencies
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  nginx \
  supervisor \
  certbot \
  python3-certbot-nginx

# Install Docker (optional, for containerized deployments)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker aiapp
```

### 4. Configure Firewall (UFW)

```bash
# Enable firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow OpenSSH

# Allow HTTP and HTTPS
sudo ufw allow 'Nginx Full'

# Enable firewall
sudo ufw enable
sudo ufw status
```

### 5. Deploy Application

```bash
# Create app directory
mkdir -p /home/aiapp/apps
cd /home/aiapp/apps

# Clone repository
git clone https://github.com/yourusername/your-ai-app.git
cd your-ai-app

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Set environment variables
sudo nano /etc/environment
# Add:
# OPENAI_API_KEY="sk-..."
# DATABASE_URL="postgresql://..."
# Add other secrets

# Or use .env file (not in version control)
cat > .env << EOF
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
EOF
```

### 6. Configure Supervisor (Process Manager)

```bash
# Create supervisor config
sudo nano /etc/supervisor/conf.d/aiapp.conf
```

Add configuration:

```ini
[program:aiapp]
directory=/home/aiapp/apps/your-ai-app
command=/home/aiapp/apps/your-ai-app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
user=aiapp
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/aiapp/err.log
stdout_logfile=/var/log/aiapp/out.log
environment=PATH="/home/aiapp/apps/your-ai-app/.venv/bin"
```

```bash
# Create log directory
sudo mkdir -p /var/log/aiapp
sudo chown aiapp:aiapp /var/log/aiapp

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start aiapp
sudo supervisorctl status
```

### 7. Configure Nginx (Reverse Proxy)

```bash
# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Create app config
sudo nano /etc/nginx/sites-available/aiapp
```

Add configuration:

```nginx
server {
    listen 80;
    server_name yourdomain.com;  # or use IP address

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running LLM requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/aiapp /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

### 8. SSL Certificate (Let's Encrypt)

```bash
# Only if you have a domain pointed to this server
sudo certbot --nginx -d yourdomain.com

# Certbot will automatically:
# - Obtain certificate
# - Configure nginx
# - Set up auto-renewal

# Test renewal
sudo certbot renew --dry-run
```

## Monitoring Setup

### Install monitoring tools

```bash
# htop for process monitoring
sudo apt install -y htop

# ncdu for disk usage
sudo apt install -y ncdu

# iotop for I/O monitoring
sudo apt install -y iotop
```

## Maintenance Commands

### Check application status
```bash
sudo supervisorctl status aiapp
```

### View application logs
```bash
sudo tail -f /var/log/aiapp/out.log
sudo tail -f /var/log/aiapp/err.log
```

### Restart application
```bash
sudo supervisorctl restart aiapp
```

### Update application
```bash
cd /home/aiapp/apps/your-ai-app
git pull
source .venv/bin/activate
uv pip install -r requirements.txt
sudo supervisorctl restart aiapp
```

### Check nginx status
```bash
sudo systemctl status nginx
sudo nginx -t  # test configuration
```

### View nginx logs
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## Security Best Practices

1. **Never use root user** for running applications
2. **Use SSH keys** instead of passwords
3. **Keep system updated**: `sudo apt update && sudo apt upgrade`
4. **Use environment variables** for secrets, never commit to git
5. **Enable firewall** (UFW) with minimal open ports
6. **Use HTTPS** always (Let's Encrypt is free)
7. **Regular backups** of data and configuration
8. **Monitor logs** for suspicious activity
9. **Rate limiting** in nginx or application
10. **Disable password authentication** for SSH

### Disable SSH password authentication

```bash
sudo nano /etc/ssh/sshd_config

# Set these values:
PasswordAuthentication no
PermitRootLogin no
PubkeyAuthentication yes

# Restart SSH
sudo systemctl restart sshd
```

## Cost Optimization

### Monitor usage
```bash
# Check disk usage
df -h
ncdu /

# Check memory usage
free -h

# Check CPU usage
htop
```

### Optimize resources
- Use appropriate VM size (don't overprovision)
- Clean up unused files and logs
- Use log rotation
- Cache expensive operations
- Consider reserved instances for long-term deployments

### Log rotation

```bash
sudo nano /etc/logrotate.d/aiapp
```

```
/var/log/aiapp/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 aiapp aiapp
    sharedscripts
    postrotate
        supervisorctl restart aiapp > /dev/null
    endscript
}
```

## Troubleshooting

### Application won't start
```bash
# Check supervisor logs
sudo supervisorctl tail aiapp stderr

# Check if port is in use
sudo netstat -tlnp | grep 8000

# Check permissions
ls -la /home/aiapp/apps/your-ai-app
```

### Can't access via domain
```bash
# Check nginx status
sudo systemctl status nginx

# Test nginx config
sudo nginx -t

# Check DNS
dig yourdomain.com

# Check firewall
sudo ufw status
```

### Out of memory
```bash
# Check memory usage
free -h

# Check which processes use most memory
ps aux --sort=-%mem | head

# Consider adding swap or upgrading VM
```

## Next Steps

After VM setup:
- Configure HTTPS with Caddy (02_https_caddy.md)
- Implement health checks (03_health_checks.py)
- Set up production logging (04_logging_config.py)
- Set up CI/CD for automated deployments (Module 4.8)

## Book Reference

- AI_eng.10 - Production deployment practices

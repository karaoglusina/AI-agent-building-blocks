# HTTPS with Caddy Reverse Proxy

Caddy is a modern web server with automatic HTTPS via Let's Encrypt. It's simpler than Nginx and perfect for AI applications.

## Why Caddy?

**Advantages over Nginx:**
- Automatic HTTPS with Let's Encrypt (zero configuration)
- Simpler configuration syntax
- Automatic certificate renewal
- HTTP/2 and HTTP/3 by default
- Built-in reverse proxy features
- No need for Certbot

**When to use Nginx instead:**
- Legacy systems with complex Nginx configs
- Need for specific Nginx modules
- Extreme performance requirements
- Team already familiar with Nginx

## Installation

### Option 1: Official Installation (Recommended)

```bash
# Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Verify installation
caddy version
```

### Option 2: Binary Download

```bash
# Download latest binary
curl -O https://caddyserver.com/api/download?os=linux&arch=amd64
chmod +x caddy
sudo mv caddy /usr/bin/

# Verify
caddy version
```

## Basic Configuration

### Caddyfile for AI Application

Create `/etc/caddy/Caddyfile`:

```bash
sudo nano /etc/caddy/Caddyfile
```

**Simple configuration (using domain):**

```caddy
yourdomain.com {
    # Reverse proxy to your FastAPI app
    reverse_proxy localhost:8000

    # Enable compression
    encode gzip

    # Access logs
    log {
        output file /var/log/caddy/access.log
    }
}
```

That's it! Caddy will automatically:
1. Get SSL certificate from Let's Encrypt
2. Redirect HTTP to HTTPS
3. Renew certificates automatically
4. Enable HTTP/2

**Advanced configuration with features:**

```caddy
yourdomain.com {
    # Reverse proxy with health checks
    reverse_proxy localhost:8000 {
        # Health check endpoint
        health_uri /health
        health_interval 30s
        health_timeout 5s

        # Load balancing (if multiple instances)
        lb_policy round_robin

        # Request headers
        header_up Host {host}
        header_up X-Real-IP {remote}
        header_up X-Forwarded-For {remote}
        header_up X-Forwarded-Proto {scheme}
    }

    # Enable compression
    encode gzip zstd

    # CORS headers (if needed for frontend)
    header {
        Access-Control-Allow-Origin "https://yourfrontend.com"
        Access-Control-Allow-Methods "GET, POST, OPTIONS"
        Access-Control-Allow-Headers "Content-Type, Authorization"
    }

    # Security headers
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
    }

    # Rate limiting (requires caddy-ratelimit plugin)
    rate_limit {
        zone dynamic {
            key {remote}
            events 100
            window 1m
        }
    }

    # Serve static files directly (if you have any)
    handle /static/* {
        root * /var/www/static
        file_server
    }

    # API routes to backend
    handle /api/* {
        reverse_proxy localhost:8000
    }

    # Health check endpoint (no logging)
    handle /health {
        reverse_proxy localhost:8000
        log {
            output discard
        }
    }

    # Access logs
    log {
        output file /var/log/caddy/access.log {
            roll_size 100mb
            roll_keep 5
            roll_keep_for 720h
        }
        format json
    }

    # Custom error pages
    handle_errors {
        @502-504 expression `{http.error.status_code} >= 502 && {http.error.status_code} <= 504`
        rewrite @502-504 /errors/5xx.html
    }
}
```

### Multiple Domains/Services

```caddy
# Main API
api.yourdomain.com {
    reverse_proxy localhost:8000
}

# Admin dashboard
admin.yourdomain.com {
    reverse_proxy localhost:8001

    # Basic auth
    basicauth {
        admin $2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG
    }
}

# Frontend
yourdomain.com {
    root * /var/www/frontend
    file_server

    # SPA routing
    try_files {path} /index.html
}
```

## Usage with IP Address (No Domain)

If you don't have a domain, use HTTP only:

```caddy
:80 {
    reverse_proxy localhost:8000
}
```

Or specify IP:

```caddy
http://YOUR_IP {
    reverse_proxy localhost:8000
}
```

**Note:** Automatic HTTPS requires a domain name. You can't get Let's Encrypt certificates for IP addresses.

## Running Caddy

### As a System Service (Recommended)

```bash
# Reload configuration
sudo systemctl reload caddy

# Start Caddy
sudo systemctl start caddy

# Enable on boot
sudo systemctl enable caddy

# Check status
sudo systemctl status caddy

# View logs
sudo journalctl -u caddy -f

# Restart
sudo systemctl restart caddy
```

### Manual Run (Testing)

```bash
# Test configuration
caddy validate --config /etc/caddy/Caddyfile

# Run in foreground (for testing)
caddy run --config /etc/caddy/Caddyfile

# Run in background
caddy start --config /etc/caddy/Caddyfile
```

## Timeouts for Long LLM Requests

AI applications may have long-running LLM requests. Configure appropriate timeouts:

```caddy
yourdomain.com {
    reverse_proxy localhost:8000 {
        # Increase timeout for LLM requests
        transport http {
            read_timeout 5m
            write_timeout 5m
            dial_timeout 30s
        }
    }
}
```

## WebSocket Support

For streaming LLM responses or real-time features:

```caddy
yourdomain.com {
    reverse_proxy localhost:8000 {
        # WebSocket support is automatic in Caddy!
        # No special configuration needed
    }

    # Or explicitly for specific paths
    @websocket {
        header Connection *Upgrade*
        header Upgrade websocket
    }
    reverse_proxy @websocket localhost:8000
}
```

## Load Balancing Multiple Instances

```caddy
yourdomain.com {
    reverse_proxy localhost:8000 localhost:8001 localhost:8002 {
        lb_policy round_robin
        lb_try_duration 2s
        lb_try_interval 500ms

        health_uri /health
        health_interval 30s
        health_timeout 5s
        health_status 2xx
    }
}
```

## Monitoring and Logs

### Log Configuration

```caddy
yourdomain.com {
    reverse_proxy localhost:8000

    # Structured JSON logs
    log {
        output file /var/log/caddy/access.log {
            roll_size 100mb
            roll_keep 5
            roll_keep_for 720h
        }
        format json
        level INFO
    }
}
```

### View Logs

```bash
# Real-time logs
sudo tail -f /var/log/caddy/access.log

# Parse JSON logs
sudo tail -f /var/log/caddy/access.log | jq .

# Filter for errors
sudo tail -f /var/log/caddy/access.log | jq 'select(.status >= 400)'

# System logs
sudo journalctl -u caddy -f
```

### Metrics Endpoint

```caddy
{
    # Enable admin API on localhost only
    admin localhost:2019
}

yourdomain.com {
    reverse_proxy localhost:8000
}
```

```bash
# Get Caddy metrics
curl localhost:2019/metrics

# Check config
curl localhost:2019/config/
```

## SSL/TLS Configuration

### Custom TLS Settings

```caddy
yourdomain.com {
    # Use staging Let's Encrypt for testing
    tls {
        ca https://acme-staging-v02.api.letsencrypt.org/directory
    }

    reverse_proxy localhost:8000
}
```

### Use Existing Certificates

```caddy
yourdomain.com {
    tls /path/to/cert.pem /path/to/key.pem
    reverse_proxy localhost:8000
}
```

### Disable HTTPS (Testing Only)

```caddy
http://yourdomain.com {
    reverse_proxy localhost:8000
}
```

## Common Issues and Solutions

### Port 80/443 Already in Use

```bash
# Check what's using the port
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :443

# Stop nginx if installed
sudo systemctl stop nginx
sudo systemctl disable nginx
```

### Certificate Errors

```bash
# Check Caddy logs
sudo journalctl -u caddy -n 50

# Test with staging certificates first
# Then switch to production

# Clear certificate cache if needed
sudo rm -rf /var/lib/caddy/.local/share/caddy/certificates
sudo systemctl restart caddy
```

### Permission Denied

```bash
# Caddy needs permission to bind to port 80/443
sudo setcap 'cap_net_bind_service=+ep' /usr/bin/caddy

# Or run as root (not recommended)
```

## Security Best Practices

### 1. Basic Authentication

```caddy
admin.yourdomain.com {
    basicauth {
        # Generate hash: caddy hash-password
        admin JDJhJDE0JFNvbWVIYXNoZWRQYXNzd29yZA==
    }

    reverse_proxy localhost:8001
}
```

Generate password hash:

```bash
caddy hash-password
# Enter password when prompted
```

### 2. IP Whitelisting

```caddy
admin.yourdomain.com {
    @allowed {
        remote_ip 203.0.113.0/24 198.51.100.42
    }

    route {
        handle @allowed {
            reverse_proxy localhost:8001
        }
        handle {
            respond "Access denied" 403
        }
    }
}
```

### 3. Rate Limiting

Install caddy-ratelimit plugin or use external service (Cloudflare, etc.)

## Comparison: Caddy vs Nginx

| Feature | Caddy | Nginx |
|---------|-------|-------|
| Auto HTTPS | Yes (built-in) | No (needs Certbot) |
| Config syntax | Simple | Complex |
| HTTP/2 | Default | Requires config |
| HTTP/3 | Built-in | Requires module |
| Certificate renewal | Automatic | Cron job needed |
| Learning curve | Easy | Moderate |
| Performance | Excellent | Excellent |
| WebSockets | Automatic | Requires config |

## Migrating from Nginx to Caddy

### Nginx config:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Equivalent Caddy config:

```caddy
yourdomain.com {
    reverse_proxy localhost:8000
}
```

That's it! 90% less configuration.

## Testing Your Setup

### 1. Test configuration

```bash
caddy validate --config /etc/caddy/Caddyfile
```

### 2. Test locally

```bash
curl -I http://localhost
curl -I https://yourdomain.com
```

### 3. Test SSL

```bash
# Check SSL certificate
echo | openssl s_client -servername yourdomain.com -connect yourdomain.com:443 2>/dev/null | openssl x509 -noout -dates

# SSL Labs test
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=yourdomain.com
```

### 4. Test performance

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Simple load test
ab -n 1000 -c 10 https://yourdomain.com/

# Better: use wrk
sudo apt install wrk
wrk -t4 -c100 -d30s https://yourdomain.com/
```

## Next Steps

After setting up Caddy:
- Implement health checks (03_health_checks.py)
- Configure production logging (04_logging_config.py)
- Set up monitoring and alerts
- Configure CI/CD for automated deployments (Module 4.8)

## Additional Resources

- Official docs: https://caddyserver.com/docs/
- Caddy Community: https://caddy.community/
- Caddyfile examples: https://github.com/caddyserver/examples

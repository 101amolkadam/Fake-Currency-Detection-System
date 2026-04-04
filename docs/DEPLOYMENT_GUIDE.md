# Deployment Guide

**Version:** 1.0.0  
**Last Updated:** April 2026  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Development Deployment](#2-development-deployment)
3. [Production Deployment](#3-production-deployment)
4. [Database Setup](#4-database-setup)
5. [Reverse Proxy (Nginx)](#5-reverse-proxy-nginx)
6. [SSL/TLS Configuration](#6-ssltls-configuration)
7. [Process Management](#7-process-management)
8. [Monitoring & Logging](#8-monitoring--logging)
9. [Backup & Recovery](#9-backup--recovery)
10. [Security Hardening](#10-security-hardening)

---

## 1. Overview

This guide covers deploying the Fake Currency Detection System in both development and production environments.

### Architecture Overview

```
                    ┌─────────────────┐
                    │   Internet      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Nginx (443)    │
                    │  SSL/TLS        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │  Frontend     │ │  Backend   │ │  Backend   │
    │  (Static)     │ │  (:8001)   │ │  (:8002)   │
    │               │ │  FastAPI   │ │  FastAPI   │
    └───────────────┘ └─────┬──────┘ └─────┬──────┘
                            │              │
                            └──────┬───────┘
                                   │
                          ┌────────▼────────┐
                          │   MySQL 8.0     │
                          │   (Local)       │
                          └─────────────────┘
```

---

## 2. Development Deployment

### Prerequisites

- Python 3.12+
- Node.js 20+
- MySQL 8.0+
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/fake-currency-detection.git
cd fake-currency-detection
```

### Step 2: Install Python Dependencies

```bash
cd backend
uv sync
```

### Step 3: Install Node Dependencies

```bash
cd ../frontend
npm install
```

### Step 4: Configure Environment

Create `backend/.env`:

```env
DATABASE_URL=mysql+pymysql://root:root@localhost:3306/fake_currency_detection
MODEL_PATH=models/xception_currency_final.h5
MAX_BASE64_SIZE=10485760
ALLOWED_ORIGINS=http://localhost:5173
ALLOWED_MIME_TYPES=image/jpeg,image/png,image/webp
```

### Step 5: Create Database

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS fake_currency_detection 
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### Step 6: Start Backend

```bash
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 7: Start Frontend

```bash
cd frontend
npm run dev
```

### Verification

- Backend health: http://127.0.0.1:8000/api/v1/health
- Frontend: http://localhost:5173
- API docs: http://127.0.0.1:8000/docs

---

## 3. Production Deployment

### Step 1: Build Frontend

```bash
cd frontend
npm run build
```

This creates `frontend/dist/` with optimized static files.

### Step 2: Configure Production Environment

Create `backend/.env`:

```env
DATABASE_URL=mysql+pymysql://currency_app:SecurePassword123@localhost:3306/fake_currency_detection
MODEL_PATH=models/xception_currency_final.h5
MAX_BASE64_SIZE=10485760
ALLOWED_ORIGINS=https://your-domain.com
ALLOWED_MIME_TYPES=image/jpeg,image/png,image/webp
```

### Step 3: Start Backend with Multiple Workers

```bash
cd backend
uv run uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log
```

### Step 4: Serve Frontend with Nginx

See Section 5 for Nginx configuration.

---

## 4. Database Setup

### 4.1 Create Dedicated User

```sql
-- Connect as root
mysql -u root -p

-- Create database
CREATE DATABASE IF NOT EXISTS fake_currency_detection 
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create dedicated user
CREATE USER 'currency_app'@'localhost' IDENTIFIED BY 'SecurePassword123';

-- Grant permissions
GRANT ALL PRIVILEGES ON fake_currency_detection.* TO 'currency_app'@'localhost';
FLUSH PRIVILEGES;
```

### 4.2 Verify Connection

```bash
mysql -u currency_app -p -e "USE fake_currency_detection; SHOW TABLES;"
```

### 4.3 Connection Pool Configuration

The application uses SQLAlchemy connection pooling:

| Parameter | Value | Description |
|-----------|-------|-------------|
| pool_size | 10 | Maximum persistent connections |
| max_overflow | 20 | Additional connections under load |
| pool_recycle | 3600 | Recycle connections after 1 hour |
| pool_pre_ping | true | Validate connections before use |

---

## 5. Reverse Proxy (Nginx)

### 5.1 Install Nginx

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx

# Windows
# Download from https://nginx.org/en/download.html
```

### 5.2 Configuration

Create `/etc/nginx/sites-available/currency-detection`:

```nginx
upstream backend {
    server 127.0.0.1:8000;
    # Add more workers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates
    ssl_certificate /etc/ssl/certs/currency-detection.crt;
    ssl_certificate_key /etc/ssl/private/currency-detection.key;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Frontend static files
    location / {
        root /path/to/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase limits for large base64 payloads
        client_max_body_size 15M;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Gzip compression
    gzip on;
    gzip_types application/json text/html text/css application/javascript;
    gzip_min_length 1000;
}
```

### 5.3 Enable Site

```bash
sudo ln -s /etc/nginx/sites-available/currency-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6. SSL/TLS Configuration

### 6.1 Obtain Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 6.2 Auto-Renewal

```bash
# Add to crontab
sudo crontab -e

# Add line:
0 0 * * 0 certbot renew --quiet --post-hook "systemctl reload nginx"
```

---

## 7. Process Management

### 7.1 Using systemd (Linux)

Create `/etc/systemd/system/currency-backend.service`:

```ini
[Unit]
Description=Fake Currency Detection Backend
After=network.target mysql.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/backend
Environment=PATH=/path/to/backend/.venv/bin
ExecStart=/path/to/backend/.venv/bin/uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable currency-backend
sudo systemctl start currency-backend
sudo systemctl status currency-backend
```

### 7.2 Using PM2 (Alternative)

```bash
npm install -g pm2

pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4" \
    --name currency-backend \
    --cwd /path/to/backend

pm2 save
pm2 startup
```

---

## 8. Monitoring & Logging

### 8.1 Backend Logs

```bash
# View logs
journalctl -u currency-backend -f

# Or with PM2
pm2 logs currency-backend
```

### 8.2 Health Monitoring

Set up automated health checks:

```bash
# Add to crontab (every 5 minutes)
*/5 * * * * curl -sf http://localhost:8000/api/v1/health || echo "Backend down!" | mail -s "Alert" admin@domain.com
```

### 8.3 Database Monitoring

```sql
-- Check connection count
SHOW STATUS LIKE 'Threads_connected';

-- Check slow queries
SHOW VARIABLES LIKE 'slow_query_log';
SET GLOBAL slow_query_log = 'ON';
```

---

## 9. Backup & Recovery

### 9.1 Database Backup

```bash
# Full backup
mysqldump -u currency_app -p fake_currency_detection > backup_$(date +%Y%m%d).sql

# Compress
gzip backup_$(date +%Y%m%d).sql

# Restore
gunzip backup_20260404.sql.gz
mysql -u currency_app -p fake_currency_detection < backup_20260404.sql
```

### 9.2 Automated Backups

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * mysqldump -u currency_app -p'SecurePassword123' fake_currency_detection | gzip > /backups/currency_$(date +\%Y\%m\%d).sql.gz

# Keep only 30 days of backups
0 3 * * * find /backups -name "currency_*.sql.gz" -mtime +30 -delete
```

### 9.3 Model Backup

```bash
# Backup model file
cp backend/models/xception_currency_final.h5 /backups/model_$(date +%Y%m%d).h5
```

---

## 10. Security Hardening

### 10.1 Firewall Configuration

```bash
# Ubuntu/Debian
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Backend only accessible via nginx
sudo ufw enable
```

### 10.2 Database Security

```sql
-- Remove anonymous users
DELETE FROM mysql.user WHERE User='';

-- Disable remote root login
UPDATE mysql.user SET Host='localhost' WHERE User='root';
FLUSH PRIVILEGES;
```

### 10.3 Environment Variables

**Never commit these files:**
- `.env`
- `*.key`
- `*.crt`

Add to `.gitignore`:
```
.env
*.key
*.crt
*.pem
backups/
```

### 10.4 Rate Limiting

Adjust in production based on expected traffic:

```python
# In main.py
limiter = Limiter(key_func=get_remote_address)
# Default: 10 requests/minute
# Increase for production:
# limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
```

---

## Quick Reference

### Start Commands

```bash
# Development
cd backend && uv run uvicorn main:app --reload
cd frontend && npm run dev

# Production
sudo systemctl start currency-backend
sudo systemctl start nginx

# Check status
sudo systemctl status currency-backend
sudo systemctl status nginx
curl http://localhost:8000/api/v1/health
```

### File Locations

| File | Path |
|------|------|
| Backend code | `/path/to/backend/` |
| Frontend build | `/path/to/frontend/dist/` |
| Model file | `/path/to/backend/models/xception_currency_final.h5` |
| Environment | `/path/to/backend/.env` |
| Nginx config | `/etc/nginx/sites-available/currency-detection` |
| Systemd service | `/etc/systemd/system/currency-backend.service` |
| Database backups | `/backups/` |

---

**Document Version:** 1.0.0  
**Last Updated:** April 2026
# System Administration Guide

**Version:** 1.0.0  
**Last Updated:** April 2026  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Daily Operations](#2-daily-operations)
3. [Monitoring & Alerting](#3-monitoring--alerting)
4. [Database Administration](#4-database-administration)
5. [Log Management](#5-log-management)
6. [Backup & Recovery](#6-backup--recovery)
7. [Troubleshooting](#7-troubleshooting)
8. [Maintenance Tasks](#8-maintenance-tasks)
9. [Scaling Guide](#9-scaling-guide)

---

## 1. System Overview

### 1.1 Components

| Component | Process | Port | Memory |
|-----------|---------|------|--------|
| Frontend | Node.js (dev) / Nginx (prod) | 5173 / 80 | 50MB / 10MB |
| Backend | Uvicorn (FastAPI) | 8000 | 350-500MB |
| Database | MySQL 8.0 | 3306 | 200-400MB |
| ML Model | Xception CNN (in-memory) | N/A | 93MB |

### 1.2 Data Flow

```
User → Frontend (React) → Backend (FastAPI) → ML Model + OpenCV → MySQL
```

---

## 2. Daily Operations

### 2.1 Starting the System

```bash
# Start MySQL
sudo systemctl start mysql

# Start Backend
cd /path/to/backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 &

# Start Frontend (production)
sudo systemctl start nginx

# Verify all components
curl -s http://127.0.0.1:8000/api/v1/health
curl -s http://localhost:80 | head -1
mysql -u currency_app -p -e "SELECT 1;"
```

### 2.2 Stopping the System

```bash
# Stop Frontend
sudo systemctl stop nginx

# Stop Backend
pkill -f uvicorn

# Stop MySQL (if needed)
sudo systemctl stop mysql
```

### 2.3 Restarting

```bash
# Graceful restart (zero downtime)
sudo systemctl reload nginx

# Backend restart (brief downtime)
sudo systemctl restart currency-backend

# Full restart
sudo systemctl restart mysql currency-backend nginx
```

### 2.4 Health Checks

```bash
# Backend health
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "database_connected": true,
#   "uptime_seconds": 3600,
#   "version": "1.0.0"
# }
```

---

## 3. Monitoring & Alerting

### 3.1 Key Metrics to Monitor

| Metric | Threshold | Alert Level |
|--------|-----------|-------------|
| Backend response time | >5s | Warning |
| Backend response time | >10s | Critical |
| Memory usage | >80% | Warning |
| Memory usage | >95% | Critical |
| Database connections | >50 | Warning |
| Database connections | >80 | Critical |
| Disk space | >80% | Warning |
| Disk space | >95% | Critical |
| Error rate (5xx) | >5% | Warning |
| Error rate (5xx) | >10% | Critical |

### 3.2 Monitoring Commands

```bash
# Memory usage
ps aux | grep -E 'uvicorn|python|mysql' | awk '{print $6/1024" MB", $11}'

# CPU usage
top -bn1 | grep -E 'uvicorn|python|mysql' | head -5

# Disk usage
df -h / | tail -1

# Database connections
mysql -u root -p -e "SHOW STATUS LIKE 'Threads_connected';"

# Active requests
ss -tnp | grep ':8000' | wc -l
```

### 3.3 Automated Health Checks

Create `/opt/scripts/health_check.sh`:

```bash
#!/bin/bash
HEALTH=$(curl -sf http://127.0.0.1:8000/api/v1/health)
if [ $? -ne 0 ]; then
    echo "Backend is DOWN!" | mail -s "ALERT: Currency Detection Backend Down" admin@domain.com
    systemctl restart currency-backend
fi
```

Add to crontab:
```bash
*/5 * * * * /opt/scripts/health_check.sh
```

---

## 4. Database Administration

### 4.1 Connection Management

```sql
-- View active connections
SELECT id, user, host, db, command, time, state 
FROM information_schema.processlist 
WHERE db = 'fake_currency_detection';

-- Kill idle connections
SELECT CONCAT('KILL ', id, ';') 
FROM information_schema.processlist 
WHERE time > 300 AND db = 'fake_currency_detection';
```

### 4.2 Table Maintenance

```sql
-- Analyze table for query optimization
ANALYZE TABLE currency_analyses;

-- Check table integrity
CHECK TABLE currency_analyses;

-- Optimize table (reclaim space)
OPTIMIZE TABLE currency_analyses;
```

### 4.3 Query Performance

```sql
-- Slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;

-- View slow queries
SELECT * FROM mysql.slow_log ORDER BY start_time DESC LIMIT 10;

-- Most frequent queries
SELECT DIGEST_TEXT, COUNT_STAR, AVG_TIMER_WAIT/1000000000 as avg_ms
FROM performance_schema.events_statements_summary_by_digest
WHERE DIGEST_TEXT LIKE '%currency_analyses%'
ORDER BY COUNT_STAR DESC LIMIT 10;
```

### 4.4 Data Cleanup

```sql
-- Delete analyses older than 90 days
DELETE FROM currency_analyses 
WHERE analyzed_at < DATE_SUB(NOW(), INTERVAL 90 DAY);

-- Check table size
SELECT 
    table_name,
    ROUND(data_length/1024/1024, 2) AS data_mb,
    ROUND(index_length/1024/1024, 2) AS index_mb,
    ROUND((data_length+index_length)/1024/1024, 2) AS total_mb
FROM information_schema.tables 
WHERE table_schema = 'fake_currency_detection';
```

---

## 5. Log Management

### 5.1 Log Locations

| Log | Location | Size |
|-----|----------|------|
| Backend | stdout/stderr (journalctl) | Rotated |
| Nginx access | `/var/log/nginx/access.log` | Rotated |
| Nginx error | `/var/log/nginx/error.log` | Rotated |
| MySQL | `/var/log/mysql/error.log` | Rotated |

### 5.2 Viewing Logs

```bash
# Backend logs
journalctl -u currency-backend -f --since "1 hour ago"

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx errors
tail -f /var/log/nginx/error.log

# MySQL errors
tail -f /var/log/mysql/error.log
```

### 5.3 Log Rotation

Create `/etc/logrotate.d/currency-backend`:

```
/var/log/currency-backend/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload currency-backend > /dev/null 2>&1 || true
    endscript
}
```

---

## 6. Backup & Recovery

### 6.1 Automated Backup Script

Create `/opt/scripts/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_USER="currency_app"
DB_PASS="SecurePassword123"
DB_NAME="fake_currency_detection"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
mysqldump -u $DB_USER -p$DB_PASS $DB_NAME | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Model backup
cp /path/to/backend/models/xception_currency_final.h5 $BACKUP_DIR/model_$DATE.h5

# Config backup
cp /path/to/backend/.env $BACKUP_DIR/env_$DATE.txt

# Keep only 30 days of backups
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.h5" -mtime +30 -delete

echo "Backup completed: $DATE"
```

### 6.2 Cron Schedule

```bash
# Daily at 2 AM
0 2 * * * /opt/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### 6.3 Restore from Backup

```bash
# Restore database
gunzip db_20260404_020000.sql.gz
mysql -u currency_app -p fake_currency_detection < db_20260404_020000.sql

# Restore model
cp model_20260404_020000.h5 /path/to/backend/models/xception_currency_final.h5
sudo systemctl restart currency-backend
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Backend won't start | Port already in use | `lsof -i :8000` and kill process |
| 500 errors | Model not loaded | Check model file exists: `ls models/*.h5` |
| Slow responses | High memory usage | Restart backend: `systemctl restart currency-backend` |
| Database errors | Connection limit reached | Increase max_connections in my.cnf |
| Frontend not loading | Nginx config error | `nginx -t` to test config |
| CORS errors | Wrong ALLOWED_ORIGINS | Update .env file and restart backend |

### 7.2 Diagnostic Commands

```bash
# Check if backend is listening
netstat -tlnp | grep 8000

# Check model file
ls -lh backend/models/xception_currency_final.h5

# Check database connection
mysql -u currency_app -p -e "USE fake_currency_detection; SELECT COUNT(*) FROM currency_analyses;"

# Check disk space
df -h

# Check memory
free -h

# Check open files
lsof -p $(pgrep -f uvicorn) | wc -l
```

### 7.3 Emergency Procedures

**Backend Crash Recovery:**
```bash
# 1. Check logs
journalctl -u currency-backend -n 100

# 2. Kill zombie processes
pkill -9 -f uvicorn

# 3. Restart
systemctl start currency-backend

# 4. Verify
curl http://127.0.0.1:8000/api/v1/health
```

**Database Corruption:**
```bash
# 1. Stop MySQL
sudo systemctl stop mysql

# 2. Check tables
myisamchk /var/lib/mysql/fake_currency_detection/*.MYI

# 3. Start MySQL
sudo systemctl start mysql

# 4. Restore from backup if needed
```

---

## 8. Maintenance Tasks

### 8.1 Daily

- [ ] Check health endpoint
- [ ] Review error logs
- [ ] Monitor disk space

### 8.2 Weekly

- [ ] Review slow query log
- [ ] Check backup integrity
- [ ] Analyze table statistics

### 8.3 Monthly

- [ ] Optimize database tables
- [ ] Review and rotate logs
- [ ] Update dependencies
- [ ] Test backup restoration

### 8.4 Quarterly

- [ ] Review and update SSL certificates
- [ ] Audit user access logs
- [ ] Performance benchmarking
- [ ] Disaster recovery drill

---

## 9. Scaling Guide

### 9.1 Vertical Scaling

| Resource | Current | Recommended for 100+ users |
|----------|---------|---------------------------|
| CPU | 2 cores | 4 cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 50 GB SSD |

### 9.2 Horizontal Scaling

**Backend:**
```nginx
upstream backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
```

**Database:**
- Master-slave replication for read scaling
- Connection pooling with ProxySQL

### 9.3 CDN for Static Files

```nginx
# Frontend assets via CDN
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

---

**Document Version:** 1.0.0  
**Last Updated:** April 2026
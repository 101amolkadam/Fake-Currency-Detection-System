# CORS Fix & Deployment Guide

## Problem
Your frontend (`validcash.netlify.app`) was trying to access your backend (`validcash.duckdns.org`), but the backend's CORS policy only allowed `http://localhost:5173`, causing the error:
```
No 'Access-Control-Allow-Origin' header is present on the requested resource
```

## What Was Fixed

### 1. **Backend CORS Configuration** (`backend/config.py`)
- Updated default `ALLOWED_ORIGINS` to include both localhost and production frontend
- Now supports: `http://localhost:5173,https://validcash.netlify.app`

### 2. **Backend `.env` File** (`backend/.env`)
- Created environment configuration file
- **IMPORTANT**: This file must be deployed to your backend server with the correct origins

### 3. **CORS Middleware Enhancement** (`backend/main.py`)
- Added explicit OPTIONS method support for preflight requests
- Added `max_age=3600` to cache preflight results
- Explicitly listed allowed methods for better security

### 4. **Frontend `.env` File** (`frontend/.env`)
- Created with correct backend URL: `https://validcash.duckdns.org/api/v1`

## Deployment Steps

### **Backend Deployment** (validcash.duckdns.org)

1. **Update your backend server** with the new code:
   ```bash
   # Pull latest changes if using git
   git pull
   
   # Or manually update these files:
   # - backend/config.py
   # - backend/main.py
   # - backend/.env (CREATE THIS FILE on server)
   ```

2. **Create `.env` file on your backend server**:
   ```env
   DATABASE_URL=mysql+pymysql://root:root@localhost:3306/fake_currency_detection
   MODEL_PATH=models/xception_currency_final.h5
   ALLOWED_ORIGINS=http://localhost:5173,https://validcash.netlify.app
   ALLOWED_MIME_TYPES=image/jpeg,image/png,image/webp
   ```

3. **Restart your backend server**:
   ```bash
   # If using systemd service
   sudo systemctl restart validcash-backend
   
   # If running manually
   cd backend
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Verify CORS is working**:
   ```bash
   # Test from terminal
   curl -X OPTIONS https://validcash.duckdns.org/api/v1/analyze \
     -H "Origin: https://validcash.netlify.app" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -v
   ```
   
   You should see these headers in the response:
   ```
   Access-Control-Allow-Origin: https://validcash.netlify.app
   Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
   Access-Control-Allow-Headers: *
   ```

### **Frontend Deployment** (validcash.netlify.app)

1. **Update environment variable on Netlify**:
   - Go to Netlify Dashboard → Your Site → Site settings → Build & deploy → Environment
   - Add: `VITE_API_BASE_URL=https://validcash.duckdns.org/api/v1`

2. **Redeploy the site**:
   ```bash
   cd frontend
   npm run build
   # Push to trigger Netlify deploy
   git add .
   git commit -m "fix: update CORS origins and backend URL"
   git push
   ```

3. **Or manually set in `frontend/.env`** (before building):
   ```env
   VITE_API_BASE_URL=https://validcash.duckdns.org/api/v1
   ```

## Testing

### 1. Test Locally
```bash
# Terminal 1 - Backend
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Open `http://localhost:5173` and try analyzing an image.

### 2. Test Production
1. Open `https://validcash.netlify.app`
2. Upload a currency image
3. Click "Analyze"
4. Should work without CORS errors

## Troubleshooting

### Still getting CORS errors?

**Check 1: Backend is using the correct `.env`**
```bash
# On your backend server, check the .env file exists and has correct content
cat /path/to/backend/.env
```

**Check 2: Backend server was restarted**
```bash
# Check if process is running
ps aux | grep uvicorn

# Restart if needed
sudo systemctl restart validcash-backend
```

**Check 3: Origins match exactly**
- `https://validcash.netlify.app` (no trailing slash)
- `http://localhost:5173` (for local dev)
- Don't mix http/https

**Check 4: Test backend directly**
```bash
# Test health endpoint
curl https://validcash.duckdns.org/api/v1/health

# Expected response:
# {"status":"healthy","model_loaded":true,"database_connected":true,...}
```

### Backend not starting?

**Check database connection:**
```bash
mysql -u root -p -e "SHOW DATABASES;"
```

**Check model file exists:**
```bash
ls -lh backend/models/xception_currency_final.h5
```

## Security Notes

1. **Don't commit `.env` files to git** (they're in `.gitignore`)
2. **Use environment variables** on your hosting platforms (Netlify, your backend server)
3. **Only allow specific origins** - don't use `*` in production
4. **Keep DATABASE_URL secure** - use strong passwords in production

## Architecture Reminder

```
Frontend (Netlify)                    Backend (Your Server)
https://validcash.netlify.app    →    https://validcash.duckdns.org
                                              ↓
                                        MySQL Database (localhost:3306)
```

The frontend is **static** (just HTML/JS/CSS) and runs in the user's browser.
The backend is **dynamic** (Python FastAPI) and must run on your server 24/7.

## Quick Checklist

- [ ] Backend `config.py` updated with production origins
- [ ] Backend `main.py` CORS middleware enhanced
- [ ] Backend `.env` file created on server
- [ ] Backend server restarted
- [ ] Frontend `.env` has correct `VITE_API_BASE_URL`
- [ ] Frontend redeployed to Netlify
- [ ] Test locally - ✓
- [ ] Test production - ✓
- [ ] Health endpoint returns healthy - ✓
- [ ] Image analysis works - ✓

## Need Help?

Run this to verify your backend is accessible:
```bash
curl -X POST https://validcash.duckdns.org/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "Origin: https://validcash.netlify.app" \
  -d '{"image":"data:image/jpeg;base64,test","source":"upload"}'
```

You should get either:
- A valid analysis response (if image is valid)
- A 400 error (if test image is invalid) - but NOT a CORS error

If you still see CORS errors, check that the `Access-Control-Allow-Origin` header is present in the response.

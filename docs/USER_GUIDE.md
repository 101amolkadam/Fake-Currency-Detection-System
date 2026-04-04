# Fake Currency Detection System
## User Guide

**Version:** 1.0.0  
**Last Updated:** April 2026  

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Starting the Application](#4-starting-the-application)
5. [Analyzing a Currency Note](#5-analyzing-a-currency-note)
   - 5.1 Uploading an Image
   - 5.2 Capturing with Camera
   - 5.3 Viewing Results
6. [Understanding the Analysis Results](#6-understanding-the-analysis-results)
   - 6.1 Overall Result
   - 6.2 Security Features Table
   - 6.3 Annotated Image
7. [Using the History Page](#7-using-the-history-page)
   - 7.1 Viewing Past Analyses
   - 7.2 Filtering Results
   - 7.3 Deleting Entries
8. [Troubleshooting](#8-troubleshooting)
9. [Frequently Asked Questions](#9-frequently-asked-questions)
10. [Best Practices](#10-best-practices)

---

## 1. Getting Started

### What This Application Does

The Fake Currency Detection System helps you verify whether an Indian currency note (₹500 or ₹2000) is genuine or counterfeit. Simply upload a photo or use your camera to capture a currency note, and the system will analyze it using artificial intelligence and computer vision.

### How It Works

1. **You provide an image**: Upload a photo or take a picture of a currency note
2. **AI analyzes the image**: A trained neural network examines the image for authenticity patterns
3. **Computer vision checks security features**: Six different security features are independently verified
4. **Results are combined**: Both analyses are combined to give you a final verdict
5. **You get detailed feedback**: See exactly which features passed or failed, with visual annotations

### What You'll See

- **Home Page**: Upload or capture your currency image
- **Results Page**: See the analysis with annotated images and detailed feature breakdown
- **History Page**: Review all your past analyses with filtering and search

---

## 2. System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | Dual-core 2.0 GHz | Quad-core 2.5 GHz |
| RAM | 4 GB | 8 GB |
| Storage | 500 MB free | 1 GB free |
| Camera | Not required | 5 MP or better |

### Software Requirements

| Component | Requirement |
|-----------|------------|
| Operating System | Windows 10/11, macOS 11+, Linux |
| Web Browser | Chrome 90+, Firefox 88+, Edge 90+, Safari 14+ |
| Internet Connection | Required for initial setup only |

### Notes

- The system runs **100% locally** — your images never leave your computer
- No internet connection needed after initial installation
- Camera access is only needed for camera capture feature
- Works on both desktop and mobile browsers

---

## 3. Installation

### Step 1: Install Python and Node.js

1. **Download Python 3.12+** from https://python.org/downloads/
   - During installation, check "Add Python to PATH"
   
2. **Download Node.js 20+** from https://nodejs.org/
   - Use the LTS (Long Term Support) version

3. **Install MySQL 8.0+** from https://dev.mysql.com/downloads/mysql/
   - Remember your root password (default: root)

### Step 2: Install uv (Python Package Manager)

```bash
# Windows
pip install uv

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Clone or Download the Project

Extract the project files to your desired location:

```
fake-currency-detection/
├── backend/          # Python backend
├── frontend/         # React frontend
├── test_images/      # Sample currency images
└── docs/             # Documentation
```

### Step 4: Install Backend Dependencies

```bash
cd backend
uv sync
```

This will download and install all required Python packages (~2GB).

### Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 6: Create the Database

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS fake_currency_detection 
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

Enter your MySQL password when prompted.

---

## 4. Starting the Application

### Step 1: Start the Backend

Open a terminal and navigate to the backend folder:

```bash
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

You should see output like:
```
[INFO] Model loaded from models/xception_currency_final.h5...
[INFO] Model loaded and warmed up successfully
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Keep this terminal open** — the backend must stay running.

### Step 2: Start the Frontend

Open a **second terminal** and navigate to the frontend folder:

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 3: Open the Application

Open your web browser and go to: **http://localhost:5173**

You should see the home page with upload and camera options.

### Stopping the Application

1. Press `Ctrl+C` in the frontend terminal
2. Press `Ctrl+C` in the backend terminal
3. Close your browser

---

## 5. Analyzing a Currency Note

### 5.1 Uploading an Image

1. On the home page, click the **"Upload Image"** tab (selected by default)

2. You have two options:
   - **Drag and Drop**: Drag a currency image file from your file explorer onto the upload area
   - **Click to Browse**: Click anywhere on the upload area to open a file picker

3. **Supported file types**: JPG, PNG, WEBP
4. **Maximum file size**: 10 MB

5. After selecting an image, you'll see a preview appear below the upload area

6. If you want to change the image, click the **✕ button** on the preview to remove it and select a different one

7. Once satisfied, click the **"🔍 Analyze Currency Note"** button

### 5.2 Capturing with Camera

1. On the home page, click the **"📷 Camera Capture"** tab

2. Click **"📸 Open Camera"**

3. Your browser will ask for camera permission — click **Allow**

4. Position the currency note in the camera view:
   - Ensure the entire note is visible
   - Keep the note flat and well-lit
   - Avoid shadows and glare

5. Click **"📸 Capture"** to take the photo

6. The captured image will appear as a preview

7. Click **"🔍 Analyze Currency Note"** to proceed

### 5.3 Viewing Results

After clicking "Analyze," the system will process your image. This typically takes **1-3 seconds**.

You'll be automatically redirected to the **Results Page**, which shows:

- **Result Badge**: Large green (REAL) or red (FAKE) indicator with confidence percentage
- **Confidence Bar**: Visual progress bar showing how confident the system is
- **Annotated Image**: Your currency image with colored overlays showing which features passed/failed
- **Analysis Table**: Detailed breakdown of all 8 analysis categories

---

## 6. Understanding the Analysis Results

### 6.1 Overall Result

The most important information is at the top of the results page:

| Indicator | Meaning |
|-----------|---------|
| ✅ REAL (green) | The note appears to be genuine |
| ❌ FAKE (red) | The note shows signs of being counterfeit |

**Confidence Percentage**: How certain the system is about its classification.

- **80-100%**: Very confident
- **60-79%**: Moderately confident
- **50-59%**: Low confidence — consider re-analyzing with a better image

### 6.2 Security Features Table

The table shows 8 rows of analysis data:

| Feature | What It Checks | Status Values |
|---------|---------------|---------------|
| **Overall Result** | Combined CNN + OpenCV score | REAL or FAKE |
| **CNN Classification** | AI neural network prediction | REAL or FAKE |
| **Watermark** | Transparency pattern in right-center area | Present or Missing |
| **Security Thread** | Thin metallic thread embedded in the note | Present or Missing |
| **Color Analysis** | Color uniformity across the note surface | Match or Mismatch |
| **Texture Quality** | Print sharpness and paper texture patterns | Normal or Abnormal |
| **Serial Number** | Unique serial number format validation | Valid or Invalid |
| **Note Dimensions** | Physical size and aspect ratio | Correct or Incorrect |

**Reading the Status:**
- ✅ **Green badge**: Feature passed verification
- ❌ **Red badge**: Feature failed verification
- The **Confidence Bar** shows how certain the system is about each feature

### 6.3 Annotated Image

The annotated image shows your currency note with colored overlays:

- **Green borders/labels**: Features that passed verification
- **Red borders/labels**: Features that failed verification
- **Yellow labels**: Features with inconclusive results

**Interactive Features:**
- **Hover over the image**: Highlights the corresponding row in the analysis table
- **Hover over the table**: Highlights the corresponding region on the image

**Legend:**
- 🟩 Green = Pass
- 🟥 Red = Fail
- 🟨 Yellow = Warning

### Understanding Confidence Scores

Each feature has a confidence score (0-100%):

| Score Range | Interpretation |
|------------|----------------|
| 80-100% | Very reliable |
| 60-79% | Moderately reliable |
| 40-59% | Low reliability |
| Below 40% | Unreliable — feature may not be detectable |

---

## 7. Using the History Page

### 7.1 Viewing Past Analyses

1. From any page, click **"📋 View Analysis History"** (on home page) or **"View History"** (on results page)

2. The history page shows all your past analyses as cards with:
   - Thumbnail image of the currency note
   - Result badge (REAL or FAKE)
   - Confidence percentage
   - Timestamp in format: `YYYY:MM:DD:hh:mm:ss_₹500`

3. Click any card to view its detailed results

### 7.2 Filtering Results

Use the filter tabs at the top to narrow down results:

- **📋 All**: Show all analyses
- **✅ Real**: Show only notes classified as REAL
- **❌ Fake**: Show only notes classified as FAKE

### 7.3 Statistics Cards

At the top of the history page, you'll see:

- **Total Analyses**: Total number of analyses performed
- **Real Notes**: Count of notes classified as REAL
- **Fake Notes**: Count of notes classified as FAKE
- **Avg Confidence**: Average confidence across all analyses

### 7.4 Pagination

If you have more than 20 analyses, use the pagination controls at the bottom:

- Click **← Previous** to go back a page
- Click **Next →** to go forward
- The page shows "Page X of Y (total analyses)"

### 7.5 Deleting Entries

To delete an analysis:

1. Find the analysis in the history list
2. Click the **🗑️ icon** on the right side
3. Confirm the deletion in the popup dialog

**Note**: Deletion cannot be undone.

---

## 8. Troubleshooting

### Problem: "Connection failed. Check if backend is running."

**Solution:**
1. Check that the backend terminal shows "Uvicorn running on http://127.0.0.1:8000"
2. If not, restart the backend: `cd backend && uv run uvicorn main:app --host 127.0.0.1 --port 8000`
3. Check that the frontend is connecting to the correct URL (http://localhost:5173)

### Problem: Camera not working

**Solution:**
1. Ensure your browser has camera permission for localhost
2. In Chrome: Click the lock icon in the address bar → Site settings → Camera → Allow
3. Try a different browser if issues persist
4. Ensure no other application is using the camera

### Problem: "Image size must be less than 10MB"

**Solution:**
1. Compress your image using an online tool
2. Take a new photo with lower resolution
3. Crop the image to focus only on the currency note

### Problem: "Only JPG, PNG, and WEBP images are supported"

**Solution:**
1. Convert your image to JPG, PNG, or WEBP format
2. You can use online converters or image editing software

### Problem: Analysis takes too long (>10 seconds)

**Solution:**
1. Close other applications to free up CPU resources
2. Use smaller images (under 1000px width)
3. Restart the backend server

### Problem: Database errors

**Solution:**
1. Ensure MySQL is running
2. Check that the database exists: `mysql -u root -p -e "SHOW DATABASES;"`
3. Verify the connection in `backend/.env`: `DATABASE_URL=mysql+pymysql://root:root@localhost:3306/fake_currency_detection`

---

## 9. Frequently Asked Questions

### Q: Is this system 100% accurate?

**A:** No system is perfect. Our testing shows 100% accuracy on genuine notes in our test set, but results may vary with different images, lighting conditions, or counterfeit quality. Always consult banking professionals for definitive authentication.

### Q: Can I use this for foreign currency?

**A:** Currently, the system is trained specifically for Indian ₹500 and ₹2000 notes. Support for other currencies requires retraining the model with appropriate datasets.

### Q: Where does my data go?

**A:** All processing happens on your local computer. Images are analyzed in memory and stored in your local MySQL database. Nothing is sent to external servers.

### Q: Can I analyze multiple notes at once?

**A:** Currently, the system processes one image at a time. Upload notes individually.

### Q: What makes a good currency image for analysis?

**A:** The best images are:
- Well-lit with even lighting
- Flat and not crumpled
- Show the entire front of the note
- In focus and not blurry
- Without shadows or glare
- High resolution (but under 10MB)

### Q: How do I improve accuracy?

**A:** 
- Use high-quality images (good lighting, focus, resolution)
- Ensure the entire currency note is visible
- Avoid shadows, glare, or partial views
- For camera capture, hold the camera steady and ensure good lighting

---

## 10. Best Practices

### For Best Results

1. **Lighting**: Use bright, even lighting. Avoid harsh shadows or direct flash glare.

2. **Positioning**: Place the currency note flat on a contrasting background (e.g., white note on dark surface).

3. **Focus**: Ensure the image is sharp and not blurry. Tap to focus if using a phone camera.

4. **Framing**: Include the entire currency note with a small border around it. Don't crop too tightly.

5. **Resolution**: Higher resolution is better, but keep files under 10MB for faster processing.

6. **Angle**: Photograph the note straight-on. Avoid angled or skewed perspectives.

7. **Cleanliness**: Ensure the note is clean and not excessively worn or torn for best analysis results.

### When to Re-analyze

- If confidence is below 60%
- If the image appears blurry or poorly lit
- If you get unexpected results
- After adjusting lighting or image quality

### Maintaining the System

1. **Regular backups**: Export your MySQL database periodically
2. **Storage management**: Delete old analyses you no longer need
3. **Updates**: Check for application updates periodically
4. **Performance monitoring**: Check the health endpoint at http://127.0.0.1:8000/api/v1/health

---

**Document Version:** 1.0.0  
**Last Updated:** April 2026
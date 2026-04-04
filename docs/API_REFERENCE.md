# API Reference Documentation

**Version:** 1.0.0  
**Base URL:** `http://127.0.0.1:8000`  
**Last Updated:** April 2026  

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Error Handling](#2-error-handling)
3. [POST /api/v1/analyze](#3-post-apiv1analyze)
4. [GET /api/v1/analyze/history](#4-get-apiv1analyzehistory)
5. [GET /api/v1/analyze/history/{id}](#5-get-apiv1analyzehistoryid)
6. [DELETE /api/v1/analyze/history/{id}](#6-delete-apiv1analyzehistoryid)
7. [GET /api/v1/health](#7-get-apiv1health)
8. [GET /api/v1/model/info](#8-get-apiv1modelinfo)
9. [Rate Limiting](#9-rate-limiting)
10. [CORS Configuration](#10-cors-configuration)

---

## 1. Authentication

The API does **not** require authentication. All endpoints are publicly accessible when running locally.

For production deployment, implement:
- API keys via FastAPI dependencies
- JWT tokens for user authentication
- Rate limiting per API key

---

## 2. Error Handling

### Standard Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid base64 format, unsupported MIME type, file too large |
| 404 | Not Found | Analysis ID does not exist |
| 422 | Unprocessable Entity | Image processing failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side processing failure |

---

## 3. POST /api/v1/analyze

Analyzes a currency note image and returns authenticity classification with detailed feature analysis.

### Request

**Content-Type:** `application/json`

**Body Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | string | Yes | Base64-encoded image with data URI prefix |
| source | string | No | "upload" or "camera" (default: "upload") |

**Image Requirements:**
- Format: `data:image/<type>;base64,<data>`
- Supported types: jpeg, png, webp
- Maximum decoded size: 10 MB

**Example Request:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAASABIAAD...",
  "source": "upload"
}
```

### Response (200 OK)

```json
{
  "id": 42,
  "result": "REAL",
  "confidence": 0.7834,
  "currency_denomination": "₹500",
  "denomination_confidence": 0.92,
  "analysis": {
    "cnn_classification": {
      "result": "REAL",
      "confidence": 0.92,
      "model": "Xception",
      "processing_time_ms": 1247
    },
    "watermark": {
      "status": "present",
      "confidence": 0.65,
      "location": {"x": 550, "y": 120, "width": 180, "height": 210},
      "ssim_score": null
    },
    "security_thread": {
      "status": "present",
      "confidence": 0.89,
      "position": "vertical",
      "coordinates": {"x_start": 210, "x_end": 216}
    },
    "color_analysis": {
      "status": "match",
      "confidence": 0.94,
      "bhattacharyya_distance": 0.06,
      "dominant_colors": null
    },
    "texture_analysis": {
      "status": "normal",
      "confidence": 0.75,
      "glcm_contrast": 0.42,
      "glcm_energy": 0.68,
      "sharpness_score": 0.81
    },
    "serial_number": {
      "status": "valid",
      "confidence": 0.9,
      "extracted_text": "2AB1234567",
      "format_valid": true
    },
    "dimensions": {
      "status": "correct",
      "confidence": 0.85,
      "aspect_ratio": 1.67,
      "expected_aspect_ratio": 1.69,
      "deviation_percent": 1.18
    }
  },
  "ensemble_score": 0.7834,
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
  "processing_time_ms": 1247,
  "timestamp": "2026-04-04T10:30:00+00:00"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| id | integer | Unique analysis ID |
| result | string | "REAL" or "FAKE" |
| confidence | float | Overall confidence (0.0-1.0) |
| currency_denomination | string\|null | Detected denomination (e.g., "₹500") |
| denomination_confidence | float\|null | Denomination detection confidence |
| analysis | object | Detailed feature analysis |
| ensemble_score | float | Raw ensemble score before thresholding |
| annotated_image | string | Base64-encoded annotated image |
| processing_time_ms | integer | Total processing time in milliseconds |
| timestamp | string | ISO 8601 timestamp of analysis |

### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid base64 image format. Expected: data:image/<type>;base64,<data>"
}
```

**400 Bad Request:**
```json
{
  "detail": "Image size (15.2MB) exceeds 10MB limit"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Analysis failed: <error details>"
}
```

---

## 4. GET /api/v1/analyze/history

Retrieves paginated list of past analyses.

### Request

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number (≥1) |
| limit | integer | 20 | Items per page (1-100) |
| filter | string | "all" | Filter by result: "all", "real", or "fake" |

**Example:**
```
GET /api/v1/analyze/history?page=2&limit=10&filter=real
```

### Response (200 OK)

```json
{
  "data": [
    {
      "id": 42,
      "result": "REAL",
      "confidence": 0.7834,
      "denomination": "₹500",
      "thumbnail": "data:image/jpeg;base64,/9j/4AAQ...",
      "analyzed_at": "2026-04-04T10:30:00+00:00"
    },
    {
      "id": 41,
      "result": "REAL",
      "confidence": 0.8912,
      "denomination": "₹2000",
      "thumbnail": "data:image/jpeg;base64,/9j/4AAQ...",
      "analyzed_at": "2026-04-04T10:25:00+00:00"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 42,
    "total_pages": 3
  }
}
```

---

## 5. GET /api/v1/analyze/history/{id}

Retrieves full details of a specific analysis.

### Request

**Path Parameters:**
- `id` (integer): Analysis ID

**Example:**
```
GET /api/v1/analyze/history/42
```

### Response (200 OK)

Same structure as POST /analyze response.

### Error Responses

**404 Not Found:**
```json
{
  "detail": "Analysis not found"
}
```

---

## 6. DELETE /api/v1/analyze/history/{id}

Deletes a specific analysis from the database.

### Request

**Path Parameters:**
- `id` (integer): Analysis ID

**Example:**
```
DELETE /api/v1/analyze/history/42
```

### Response (200 OK)

```json
{
  "message": "Analysis deleted successfully"
}
```

### Error Responses

**404 Not Found:**
```json
{
  "detail": "Analysis not found"
}
```

---

## 7. GET /api/v1/health

Health check endpoint for monitoring.

### Response (200 OK)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| status | string | Overall health status ("healthy" or "degraded") |
| model_loaded | boolean | Whether the Xception model is loaded in memory |
| database_connected | boolean | Whether MySQL connection is active |
| uptime_seconds | integer | Server uptime in seconds |
| version | string | API version |

---

## 8. GET /api/v1/model/info

Returns information about the loaded ML model.

### Response (200 OK)

```json
{
  "architecture": "Xception (ImageNet pretrained + fine-tuned)",
  "status": "loaded",
  "supported_denominations": ["₹10", "₹20", "₹50", "₹100", "₹200", "₹500", "₹2000"],
  "fallback_mode": false
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| architecture | string | Model architecture description |
| status | string | "loaded" or "not_loaded" |
| supported_denominations | array | List of supported currency denominations |
| fallback_mode | boolean | true if running in OpenCV-only mode (no CNN) |

---

## 9. Rate Limiting

The API implements rate limiting using `slowapi`:

| Limit | Scope |
|-------|-------|
| 10 requests | Per IP address per minute |

**Rate Limit Exceeded Response (429):**
```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests. Try again later."
  }
}
```

---

## 10. CORS Configuration

**Allowed Origins:**
- Development: `http://localhost:5173`
- Production: Configure via `ALLOWED_ORIGINS` environment variable

**Allowed Methods:**
- GET, POST, DELETE, OPTIONS

**Allowed Headers:**
- All headers (*)

**Credentials:**
- Enabled (cookies and auth headers allowed)

---

**Document Version:** 1.0.0  
**Last Updated:** April 2026
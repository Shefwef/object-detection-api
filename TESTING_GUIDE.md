# 🚀 Object Detection API - Swagger UI Testing Guide

## Access the API Documentation
**URL:** http://localhost:8000/docs

This opens an interactive Swagger UI where you can test all endpoints with real requests!

---

## 📋 Endpoint Overview

### **1. Health Check** (GET /health)
Shows status of all models
- **Expected:** All models show `is_loaded: False` initially (lazy loaded on first use)
- **Time:** ~0.1 seconds

**Try it:**
1. Find `GET /health` (green button)
2. Click **"Try it out"** → **"Execute"**

---

### **2. YOLOv8 Detection** (POST /api/v1/yolo/detect) ✅ WORKING
Real-time object detection on 80 COCO classes

**Parameters:**
- `file`: Image file (jpg/png)
- `confidence`: Detection threshold (0-1, default 0.25)
- `iou_threshold`: NMS threshold (default 0.45)
- `max_detections`: Max objects to return (default 300)

**Expected Response:**
```json
{
  "model": "yolov8",
  "detections": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class_id": 5,
      "class_name": "bus"
    }
  ],
  "detection_count": 6,
  "inference_time_ms": 2150
}
```

**Try it:**
1. Find `POST /api/v1/yolo/detect`
2. Click **"Try it out"**
3. Under `file`, click **"Choose File"**
4. Select: `C:\Users\Shefayat\Desktop\Resources\CV\Objection-Detection-Model\sample_images\bus.jpg`
5. Leave `confidence=0.25`
6. Click **"Execute"**

**What you'll see:**
- First request: ~5-6 seconds (downloading + loading model)
- Subsequent: ~2 seconds
- **Results:** 6 objects detected (bus, persons, etc.)
- **Scores:** Bus 87%, Person 86%, etc.

---

### **3. Grounding DINO Detection** (POST /api/v1/grounding-dino/detect) ✅ FIXED!
**Open-set detection** - detect ANY object by text description

**Parameters:**
- `file`: Image file
- `text_prompt`: What to find (separate items with periods: "bus. person. car.")
- `box_threshold`: Detection confidence (0-1, default 0.35)
- `text_threshold`: Text matching confidence (0-1, default 0.25)

**Expected Response:**
```json
{
  "model": "grounding_dino",
  "detections": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.92,
      "label": "bus",
      "text_prompt": "bus. person. car."
    }
  ],
  "count": 5,
  "text_prompt": "bus. person. car."
}
```

**Try it:**
1. Find `POST /api/v1/grounding-dino/detect`
2. Click **"Try it out"**
3. Choose file: `bus.jpg`
4. Enter text prompt: `"bus. person. car."`
5. Click **"Execute"**

**What you'll see:**
- First request: ~15 seconds (downloading HuggingFace model)
- Results: Bus, persons detected by TEXT MATCHING!
- **This is powerful:** You describe what to find, it finds it automatically

---

### **4. SAM Segmentation** (POST /api/v1/sam/segment-auto) ✅ WORKING
**Segment Anything** - automatic mask generation

**Parameters:**
- `file`: Image file
- `points_per_side`: Grid resolution (default 32, for auto-mode)
- `pred_iou_thresh`: Prediction IOU threshold (default 0.88)

**Expected Response:**
```json
{
  "model": "sam",
  "masks": [
    {
      "id": 0,
      "area": 1234,
      "bbox": [x1, y1, x2, y2],
      "stability_score": 0.95,
      "crop_box": [x1, y1, x2, y2]
    }
  ],
  "mask_count": 8,
  "image_shape": [1080, 810],
  "inference_time_ms": 150000
}
```

**Try it:**
1. Find `POST /api/v1/sam/segment-auto`
2. Click **"Try it out"**
3. Choose file: `bus.jpg`
4. Click **"Execute"**

**What you'll see:**
- ⏳ **SLOW** (2-3 minutes first time - model is large)
- Results: 8+ segmentation masks
- Each mask is a pixel-perfect boundary around objects
- **Super powerful:** No annotations needed, finds everything!

---

### **5. Combined Pipeline** (POST /api/v1/pipeline/detect-and-segment) 🔥 NEW!
**Grounding DINO + SAM** - Text-based segmentation

**Parameters:**
- `file`: Image file  
- `text_prompt`: What to segment
- `box_threshold`: Detection threshold (0.35)

**How it works:**
1. Grounding DINO detects object based on text
2. SAM segments the detected objects with precision
3. Returns masks for matched objects

**Expected Response:**
```json
{
  "detection_model": "grounding_dino",
  "segmentation_model": "sam",
  "text_prompt": "bus",
  "detections": [...],
  "segments": [
    {
      "detection_id": 0,
      "label": "bus",
      "mask": {...},
      "confidence": 0.92
    }
  ],
  "detection_count": 1,
  "segment_count": 1
}
```

**Try it:**
1. Find `POST /api/v1/pipeline/detect-and-segment`
2. Click **"Try it out"**
3. Choose file: `bus.jpg`
4. Enter prompt: `"bus"`
5. Click **"Execute"**

**What you'll see:**
- Slow (~2-3 minutes - running both models)
- Results: Bus detected, then segmented
- **Amazing:** Text → Perfect segmentation mask!

---

## 🎯 Recommended Testing Order

1. **Start with Health Check** (2 seconds)
   - Verify API is responding

2. **Test YOLO** (5 seconds first, 2 seconds after)
   - Fastest model, instant feedback

3. **Test Grounding DINO** (15 seconds first)
   - Amazing text-based detection
   - Try different prompts: "person. dog." or "vehicle"

4. **Test SAM** (2-3 minutes 😅)
   - Worth the wait! Pixel-perfect masks

5. **Test Pipeline** (2-3 minutes)
   - Best of both: text-based segmentation

---

## 📊 Model Performance Summary

| Model | Speed | Features | First Request |
|-------|-------|----------|---|
| **YOLO** | ⚡ 2-5s | Real-time, 80 classes | 5s (loads model) |
| **Grounding DINO** | 🚀 ~1-2s | Text-based open-set | 15s (HF download) |
| **SAM** | 🐢 2-3min | Pixel-perfect masks | 3min (large model) |
| **Pipeline** | 🐢 2-3min | Text → Segmentation | 3min (combines both) |

---

## 🆘 Troubleshooting

**"Connection refused"**
- Check server is running: `curl http://localhost:8000/health`

**"422 Unprocessable Entity"**  
- Wrong parameter format
- File upload issue
- Try using the file picker instead of manual input

**"500 Internal Server Error"**
- Model loading issue
- Check terminal logs for detailed error
- Try running health check first

**Slow SAM (2-3 minutes)**
- ✅ Normal! SAM is a large model (375MB)
- Subsequent requests are faster due to GPU caching
- First time downloads model checkpoint

**Want lighter SAM?**
- Can switch from `vit_b` to `vit_l` or `vit_h` in config
- (Currently using `vit_b` - best balance of speed/accuracy)

---

## 💡 Pro Tips

1. **Copy Response as cURL:**
   - Swagger shows request as cURL in browser dev tools
   - Useful for automation scripts

2. **Test with Your Own Images:**
   - Download image to `sample_images/`
   - Use file picker to select

3. **Try Different Prompts:**
   - Grounding DINO: "dog. cat. person."
   - "damaged area on car"
   - "people wearing red"

4. **Combine Models:**
   - Use Pipeline for intelligent workflows
   - DINO for detection, SAM for segmentation

5. **Monitor Performance:**
   - Check "inference_time_ms" in responses
   - Good for production optimization

---

## 🚀 Next Steps (After Testing)

- ✅ Models work!
- ⏭️ Dockerize the app
- ⏭️ Deploy to Azure
- ⏭️ Scale with multiple workers

---

**Ready? Open http://localhost:8000/docs now!** 🎉

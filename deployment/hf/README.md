---
title: Object Detection Studio
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: YOLOv8, Detectron2, Grounding DINO, and SAM behind one FastAPI.
---

# Object Detection Studio — Backend

FastAPI service that unifies **YOLOv8**, **Detectron2 (Mask R-CNN)**,
**Grounding DINO**, and **SAM** behind a single REST API, plus a
Grounding DINO → SAM open-vocabulary segmentation pipeline, Grad-CAM
explainability, and a live per-model metrics dashboard.

- Interactive docs: [`/docs`](/docs)
- Health probe: [`/health`](/health)
- Metrics: [`/api/v1/metrics/summary`](/api/v1/metrics/summary)

## Endpoints

| Path | Purpose |
|---|---|
| `POST /api/v1/yolo/detect` | Real-time object detection |
| `POST /api/v1/yolo/detect-base64` | Same, base64 (webcam) |
| `POST /api/v1/detectron2/detect` | Instance segmentation |
| `POST /api/v1/grounding-dino/detect` | Open-set text-driven detection |
| `POST /api/v1/sam/segment-*` | Universal segmentation (auto/points/boxes) |
| `POST /api/v1/pipeline/detect-and-segment` | Text → detect → segment |
| `POST /api/v1/explain/gradcam` | Grad-CAM heatmap for a YOLO detection |
| `GET /api/v1/metrics/summary` | Per-model latency + throughput |

## Source

Full source, architecture diagrams, tests, and deployment scripts:
[github.com/Shefwef/object-detection-api](https://github.com/Shefwef/object-detection-api)

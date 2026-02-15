# Comparative Analysis of Modern Object Detection & Segmentation Models: A Unified Inference Platform

> **Academic Research & Development Project**
> Exploring real-time object detection and segmentation architectures through a production-grade unified inference system.

---

## Abstract

This project presents a **comparative study and unified deployment framework** for four state-of-the-art computer vision models: **YOLOv8** (single-stage detection), **Detectron2** (two-stage instance segmentation), **Grounding DINO** (open-set language-grounded detection), and **SAM** (foundation model for universal segmentation). The system exposes all models through a standardized REST API, enabling direct comparison of inference characteristics, accuracy trade-offs, and deployment considerations across architectures.

A key contribution is the implementation of a **Grounding DINO + SAM pipeline** that achieves open-vocabulary instance segmentation — detecting and segmenting arbitrary objects from natural language descriptions without model retraining.

The platform is containerized with Docker and includes Infrastructure-as-Code templates for cloud deployment on AWS (ECS Fargate) and Kubernetes, demonstrating MLOps best practices for serving computer vision models in production.

---

## Research Objectives

1. **Architectural Comparison**: Understand the design differences between single-stage (YOLO), two-stage (Detectron2), language-grounded (Grounding DINO), and foundation (SAM) model architectures.
2. **Pipeline Composition**: Investigate how combining detection models with segmentation foundation models (Grounding DINO → SAM) enables zero-shot instance segmentation.
3. **Production Deployment**: Study the engineering requirements for serving large CV models via containerized microservices with proper health monitoring, autoscaling, and GPU/CPU flexibility.

--- 

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                        │
│                  (REST API + Swagger)                     │
├──────────┬──────────┬───────────────┬───────────────────┤
│  YOLOv8  │Detectron2│ Grounding DINO│   SAM (Segment    │
│ (Real-   │ (Instance│ (Open-Set     │   Anything Model) │
│  time    │  Seg &   │  Detection    │                   │
│  Det.)   │  Det.)   │  w/ Text)     │                   │
├──────────┴──────────┴───────────────┴───────────────────┤
│              Model Service Layer                         │
│         (Lazy Loading + Caching + GPU Support)           │
├─────────────────────────────────────────────────────────┤
│              Docker Container (CUDA-ready)                │
├─────────────────────────────────────────────────────────┤
│         AWS ECS / Kubernetes Deployment                   │
└─────────────────────────────────────────────────────────┘
```

---

## Models Studied

### 1. YOLOv8 (You Only Look Once v8)
| Aspect | Detail |
|--------|--------|
| **Type** | Single-stage, anchor-free detector |
| **Architecture** | CSPDarknet backbone → PANet neck → Detection head |
| **Strength** | Real-time speed (100+ FPS on GPU) |
| **Pre-training** | 80 COCO classes |
| **Key Innovation** | Processes entire image in one forward pass |

### 2. Detectron2 (Mask R-CNN)
| Aspect | Detail |
|--------|--------|
| **Type** | Two-stage detector with mask branch |
| **Architecture** | ResNet-50 + FPN → RPN → ROI Heads + Mask Head |
| **Strength** | High-quality instance segmentation |
| **Pre-training** | COCO instance segmentation |
| **Key Innovation** | Separate region proposal and classification stages |

### 3. Grounding DINO
| Aspect | Detail |
|--------|--------|
| **Type** | Multi-modal, language-grounded detector |
| **Architecture** | Swin Transformer (image) + BERT (text) → Cross-modal decoder |
| **Strength** | Detect ANY object from text description |
| **Pre-training** | Large-scale image-text pairs |
| **Key Innovation** | Open-set detection — no fixed class vocabulary |

### 4. SAM (Segment Anything Model)
| Aspect | Detail |
|--------|--------|
| **Type** | Foundation model for segmentation |
| **Architecture** | ViT image encoder + Prompt encoder + Mask decoder |
| **Strength** | Universal segmentation from any prompt type |
| **Pre-training** | SA-1B dataset (11M images, 1B+ masks) |
| **Key Innovation** | Compute image embedding once, prompt many times |

---

## Key Pipeline: Grounding DINO + SAM

The combined pipeline achieves **open-vocabulary instance segmentation**:

```
Text: "red car on the street"
          ↓
  ┌─────────────────┐
  │  Grounding DINO  │  ← Text + Image → Bounding boxes
  └────────┬────────┘
           ↓ boxes
  ┌─────────────────┐
  │       SAM        │  ← Boxes as prompts → Pixel masks
  └────────┬────────┘
           ↓
  Detected + Segmented regions with natural language control
```

**Why this matters**: Traditional models detect fixed classes. This pipeline lets you describe *anything* and get precise segmentation — no retraining needed when requirements change.

---

## Quick Start

### Local Development

```bash
git clone <repo-url>
cd Objection-Detection-Model
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# GPU build (NVIDIA Container Toolkit required)
docker-compose up --build

# CPU-only build
docker-compose --profile cpu up --build
```

### Access
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## API Endpoints

| Endpoint | Method | Model | Description |
|----------|--------|-------|-------------|
| `/api/v1/yolo/detect` | POST | YOLOv8 | Real-time object detection |
| `/api/v1/yolo/detect-visualize` | POST | YOLOv8 | Detection + annotated image |
| `/api/v1/detectron2/detect` | POST | Detectron2 | Instance segmentation |
| `/api/v1/detectron2/detect-visualize` | POST | Detectron2 | Segmentation + visualization |
| `/api/v1/grounding-dino/detect` | POST | G-DINO | Open-set text-based detection |
| `/api/v1/sam/segment-auto` | POST | SAM | Auto-segment everything |
| `/api/v1/sam/segment-points` | POST | SAM | Segment from click points |
| `/api/v1/sam/segment-boxes` | POST | SAM | Segment from bounding boxes |
| `/api/v1/pipeline/detect-and-segment` | POST | G-DINO+SAM | Text → Detect → Segment |
| `/health` | GET | — | System health & model status |

### Example: Grounding DINO + SAM Pipeline

```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/detect-and-segment" \
  -F "file=@test_image.jpg" \
  -F "text_prompt=person wearing helmet"
```

---

## Deployment

### AWS (ECS Fargate)

```bash
cd deployment/aws
./ecr-push.sh          # Push image to ECR
./deploy.sh            # Deploy via CloudFormation
```

Infrastructure: VPC → ALB → ECS Fargate Service. See [deployment/aws/README.md](deployment/aws/README.md).

### Kubernetes

```bash
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml
kubectl apply -f deployment/k8s/hpa.yaml        # Auto-scaling
```

Includes readiness/liveness probes, HPA (autoscaling on CPU/memory), and rolling update strategy.

---

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Centralized configuration
│   ├── models/
│   │   ├── base_model.py       # Abstract interface (Strategy Pattern)
│   │   ├── yolo_model.py       # YOLOv8 wrapper
│   │   ├── detectron2_model.py # Detectron2 Mask R-CNN wrapper
│   │   ├── grounding_dino.py   # Grounding DINO wrapper
│   │   └── sam_model.py        # SAM wrapper
│   ├── routers/                # API endpoint definitions
│   ├── schemas/                # Pydantic request/response models
│   └── utils/                  # Image processing & visualization
├── tests/                      # pytest test suite
├── deployment/
│   ├── aws/                    # ECR + ECS Fargate + CloudFormation
│   └── k8s/                    # Kubernetes manifests + HPA
├── Dockerfile                  # Multi-stage CUDA-ready build
├── docker-compose.yml          # Local development orchestration
└── requirements.txt
```

---

## Engineering Practices

| Practice | Implementation |
|----------|---------------|
| **Model Abstraction** | Strategy Pattern via `BaseDetectionModel` ABC |
| **Lazy Loading** | Models load on first request, not at startup |
| **API Versioning** | All routes under `/api/v1/` prefix |
| **Input Validation** | Pydantic schemas with type constraints |
| **Health Monitoring** | `/health` endpoint reports per-model status |
| **Container Security** | Non-root user, multi-stage build, `.dockerignore` |
| **Auto-scaling** | K8s HPA on CPU/memory; ECS desired count |
| **IaC** | CloudFormation for AWS; declarative K8s manifests |

---

## Testing

```bash
pytest          # Run all tests
pytest -v       # Verbose output
```

---

## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). *Detectron2*. https://github.com/facebookresearch/detectron2
- Liu, S., Zeng, Z., et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. arXiv:2303.05499
- Kirillov, A., Mintun, E., et al. (2023). *Segment Anything*. arXiv:2304.02643

---

## Author

**Shefayat E Shams Adib**
Islamic University of Technology (IUT), Dhaka
shefayatadib@iut-dhaka.edu

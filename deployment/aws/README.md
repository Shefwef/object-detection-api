# AWS Deployment Guide

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   Client     │────▶│   ALB        │────▶│   ECS Fargate        │
│   (Browser/  │     │ (Application │     │   ┌────────────────┐ │
│    curl)     │     │  Load        │     │   │ CV Detection   │ │
│              │     │  Balancer)   │     │   │ Container      │ │
└──────────────┘     └──────────────┘     │   │ (port 8000)    │ │
                                          │   └────────────────┘ │
                                          └──────────────────────┘
                                                     │
                                          ┌──────────────────────┐
                                          │   ECR (Container     │
                                          │   Registry)          │
                                          └──────────────────────┘
```

## Prerequisites

1. [AWS CLI](https://aws.amazon.com/cli/) installed and configured
2. Docker installed locally
3. AWS account with permissions for ECR, ECS, VPC, ALB

## Deployment Steps

### 1. Push Docker Image to ECR

```bash
cd deployment/aws
chmod +x ecr-push.sh
./ecr-push.sh
```

### 2. Deploy Infrastructure with CloudFormation

```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Verify Deployment

```bash
# Get the ALB URL from CloudFormation outputs
aws cloudformation describe-stacks \
  --stack-name cv-detection-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
  --output text

# Test health endpoint
curl http://<ALB-URL>/health
```

## Environment Variables

Set these in the ECS task definition or pass via `deploy.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| DEVICE | cpu | Use `cpu` for Fargate, `cuda` for GPU instances |
| YOLO_MODEL_NAME | yolov8n.pt | YOLO model variant |
| SAM_MODEL_TYPE | vit_b | SAM model size |
| LOG_LEVEL | INFO | Logging verbosity |

## Costs (Approximate)

| Resource | Config | Monthly Cost |
|----------|--------|-------------|
| ECS Fargate | 2 vCPU, 8GB RAM | ~$65 |
| ALB | Standard | ~$22 |
| ECR | 5GB storage | ~$0.50 |
| **Total** | | **~$88/month** |

For GPU workloads, use EC2 launch type with `g4dn.xlarge` (~$380/month).

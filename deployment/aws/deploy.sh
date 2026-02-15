#!/bin/bash
# ════════════════════════════════════════════════════════════
#  Deploy CV Detection Platform to AWS ECS (Fargate)
# ════════════════════════════════════════════════════════════
#
#  This script:
#    1. Pushes image to ECR (if not done already)
#    2. Deploys CloudFormation stack (VPC, ALB, ECS, etc.)
#    3. Waits for deployment and prints the endpoint URL
#
#  Usage:
#    ./deploy.sh
#    ./deploy.sh ap-southeast-1 cv-detection-platform latest
# ════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ────────────────────────────────────────
AWS_REGION="${1:-ap-southeast-1}"
ECR_REPO_NAME="${2:-cv-detection-platform}"
IMAGE_TAG="${3:-latest}"
STACK_NAME="cv-detection-stack"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Deploying CV Detection Platform to AWS              ║"
echo "║  Stack:  ${STACK_NAME}                               ║"
echo "║  Region: ${AWS_REGION}                               ║"
echo "║  Image:  ${FULL_IMAGE}                               ║"
echo "╚══════════════════════════════════════════════════════╝"

# ─── Step 1: Ensure image exists in ECR ──────────────────
echo "[1/3] Checking ECR image..."
if ! aws ecr describe-images --repository-name "${ECR_REPO_NAME}" --image-ids imageTag="${IMAGE_TAG}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "  Image not found in ECR. Building and pushing..."
    bash "${SCRIPT_DIR}/ecr-push.sh" "${AWS_REGION}" "${ECR_REPO_NAME}" "${IMAGE_TAG}"
fi

# ─── Step 2: Deploy CloudFormation ───────────────────────
echo "[2/3] Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/cloudformation.yaml" \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ContainerImage="${FULL_IMAGE}" \
        ContainerPort=8000 \
        TaskCpu=2048 \
        TaskMemory=8192 \
        DesiredCount=1 \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset

# ─── Step 3: Get endpoint URL ────────────────────────────
echo "[3/3] Retrieving endpoint..."
ALB_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
    --output text)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ Deployment Complete                              ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  API URL:    http://${ALB_URL}                       ║"
echo "║  Swagger:    http://${ALB_URL}/docs                  ║"
echo "║  Health:     http://${ALB_URL}/health                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Test with:"
echo "  curl http://${ALB_URL}/health"
echo "  curl -X POST http://${ALB_URL}/api/v1/yolo/detect -F 'file=@image.jpg'"

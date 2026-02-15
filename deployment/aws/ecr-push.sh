#!/bin/bash
# ════════════════════════════════════════════════════════════
#  Push Docker image to Amazon ECR (Elastic Container Registry)
# ════════════════════════════════════════════════════════════
#
#  Usage:
#    ./ecr-push.sh                          # Uses defaults
#    ./ecr-push.sh my-region my-repo 1.0.0  # Custom values
#
#  Prerequisites:
#    - AWS CLI configured (aws configure)
#    - Docker running
# ════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ────────────────────────────────────────
AWS_REGION="${1:-ap-southeast-1}"          # Singapore (closest to Dhaka)
ECR_REPO_NAME="${2:-cv-detection-platform}"
IMAGE_TAG="${3:-latest}"

# ─── Get AWS Account ID ──────────────────────────────────
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Pushing to ECR                                      ║"
echo "║  Account:  ${AWS_ACCOUNT_ID}                         ║"
echo "║  Region:   ${AWS_REGION}                             ║"
echo "║  Image:    ${FULL_IMAGE}                             ║"
echo "╚══════════════════════════════════════════════════════╝"

# ─── Step 1: Create ECR repository (if not exists) ───────
echo "[1/4] Creating ECR repository..."
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256

# ─── Step 2: Authenticate Docker with ECR ────────────────
echo "[2/4] Authenticating Docker with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

# ─── Step 3: Build Docker image ─────────────────────────
echo "[3/4] Building Docker image..."
cd "$(dirname "$0")/../.."  # Navigate to project root
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# ─── Step 4: Tag and push ───────────────────────────────
echo "[4/4] Pushing to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo ""
echo "✅ Successfully pushed: ${FULL_IMAGE}"
echo ""
echo "To deploy to ECS, run:"
echo "  ./deploy.sh ${AWS_REGION} ${ECR_REPO_NAME} ${IMAGE_TAG}"

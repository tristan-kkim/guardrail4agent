#!/bin/bash
# ECR 레포지토리 생성 및 Docker 이미지 빌드/푸시
# 사용법: bash scripts/build_push_ecr.sh <aws-account-id> <region>

set -e

ACCOUNT_ID=${1:-$(aws sts get-caller-identity --query Account --output text)}
REGION=${2:-ap-northeast-2}
REPO_NAME="guardrail4agent"
TAG="latest"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

echo "Account: ${ACCOUNT_ID}"
echo "Region:  ${REGION}"
echo "Image:   ${IMAGE_URI}"

# ECR 레포지토리 생성 (이미 존재하면 무시)
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

# ECR 로그인
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# HuggingFace DLC 베이스 이미지 pull 권한 (us-east-1 기준 공개 이미지 사용 시 불필요)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# 빌드 및 푸시
docker build -t "${REPO_NAME}:${TAG}" .
docker tag "${REPO_NAME}:${TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "이미지 푸시 완료: ${IMAGE_URI}"
echo ""
echo "configs/sagemaker.yaml의 docker_image를 아래로 업데이트하세요:"
echo "  docker_image: ${IMAGE_URI}"

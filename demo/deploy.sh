#!/bin/bash
# AWS EC2 배포 스크립트
# 사전 준비: EC2 (Ubuntu 22.04), 포트 80 인바운드 허용, Docker 설치됨
#
# 사용법:
#   chmod +x deploy.sh
#   ./deploy.sh <EC2_PUBLIC_IP> <PEM_KEY_PATH>
#
# 예시:
#   ./deploy.sh 13.125.x.x ~/.ssh/my-key.pem

set -e

EC2_IP=${1:?"EC2 IP를 첫 번째 인수로 입력하세요"}
PEM=${2:?"PEM 키 경로를 두 번째 인수로 입력하세요"}
REMOTE="ubuntu@${EC2_IP}"

echo "▶ EC2에 Docker 설치 확인..."
ssh -i "$PEM" -o StrictHostKeyChecking=no "$REMOTE" \
  "command -v docker || (sudo apt-get update -q && sudo apt-get install -y -q docker.io docker-compose-plugin && sudo usermod -aG docker ubuntu)"

echo "▶ demo/ 폴더 업로드..."
rsync -avz --exclude '__pycache__' -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/" "${REMOTE}:/home/ubuntu/guardrail-demo/"

echo "▶ .env 파일 업로드..."
scp -i "$PEM" -o StrictHostKeyChecking=no \
  "$(dirname "$0")/.env" "${REMOTE}:/home/ubuntu/guardrail-demo/.env"

echo "▶ Docker Compose 빌드 & 실행..."
ssh -i "$PEM" -o StrictHostKeyChecking=no "$REMOTE" \
  "cd /home/ubuntu/guardrail-demo && sudo docker compose up -d --build"

echo ""
echo "✅ 배포 완료!"
echo "   접속 주소: http://${EC2_IP}"
echo "   로그 확인: ssh -i $PEM $REMOTE 'sudo docker compose -f /home/ubuntu/guardrail-demo/docker-compose.yml logs -f'"

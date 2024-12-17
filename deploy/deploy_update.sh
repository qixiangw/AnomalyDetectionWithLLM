#!/bin/bash

set -e

# 配置
AWS_ACCOUNT_ID="249517808360"
REGION="us-west-2"
FUNCTION_NAME="anomaly-detection-with-iamge"  # Fixed typo in name
IMAGE_NAME="lambda-anomaly-detection"
VERSION=$(date +%Y%m%d-%H%M%S)  # 使用时间戳作为版本号

# 日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"  # Fixed string interpolation
}

# 错误处理
handle_error() {
    log "Error occurred in script at line $1"  # Fixed string interpolation
    exit 1
}

trap 'handle_error $LINENO' ERR

# 主要步骤
log "Starting update process..."

# 检查 ECR 仓库是否存在，如果不存在则创建
if ! aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${REGION} 2>/dev/null; then
    log "Creating ECR repository..."
    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION}
fi

# 登录 ECR
log "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 构建镜像
log "Building Docker image..."
# 使用 buildx 来确保正确的平台
docker buildx create --use
docker buildx build --platform linux/amd64 -t ${IMAGE_NAME}:${VERSION} --load .

# 标记镜像
log "Tagging image..."
docker tag ${IMAGE_NAME}:${VERSION} ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${VERSION}

# 推送镜像
log "Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${VERSION}

# 更新 Lambda
log "Updating Lambda function..."
aws lambda update-function-code \
    --region ${REGION} \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${VERSION}

# 等待更新完成
log "Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name ${FUNCTION_NAME}

# 清理
# log "Cleaning up old images..."
# docker image prune -f

log "Update completed successfully!"

#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy.sh — Build, push, and deploy to Cloud Run (local/manual alternative to CI)
#
# Prerequisites: gcloud CLI authenticated, Docker running, Terraform installed.
#
# Usage:
#   export PROJECT_ID="shoot-ai-poc"
#   export REGION="us-central1"
#   export ENVIRONMENT="dev"
#   export GEMINI_API_KEY="your-key"
#   export TF_VAR_api_keys='["your-dev-key"]'   # Terraform list(string) format
#   bash infra/scripts/deploy.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
APP_NAME="shoot-ai"
TFSTATE_BUCKET="${PROJECT_ID}-tfstate"

NAME_PREFIX="${APP_NAME}-${ENVIRONMENT}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${NAME_PREFIX}"
IMAGE="${REGISTRY}/${APP_NAME}"
SHA="$(git rev-parse --short HEAD)"

echo "==> Deploy: ${IMAGE}:${SHA} → Cloud Run ${NAME_PREFIX}"

# 1. Configure Docker for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 2. Build image
echo "==> Building image..."
docker build --platform linux/amd64 -t "${IMAGE}:${SHA}" -t "${IMAGE}:latest" .

# 3. Push to Artifact Registry
echo "==> Pushing image..."
docker push "${IMAGE}:${SHA}"
docker push "${IMAGE}:latest"

# 4. Terraform apply
echo "==> Applying Terraform..."
cd infra/terraform
terraform init -input=false \
  -backend-config="bucket=${TFSTATE_BUCKET}" \
  -backend-config="prefix=terraform/state/${PROJECT_ID}"

terraform apply -auto-approve \
  -var-file="environments/${ENVIRONMENT}/terraform.tfvars" \
  -var="gemini_api_key=${GEMINI_API_KEY}" \
  -var="image_tag=${SHA}"
# api_keys is passed via TF_VAR_api_keys env var (list format: '["key1","key2"]')

SERVICE_URL="$(terraform output -raw service_url)"
cd ../..

# 5. Smoke test
echo "==> Smoke test: ${SERVICE_URL}/health"
curl --retry 5 --retry-delay 3 --fail "${SERVICE_URL}/health"

echo ""
echo "==> Deployed: ${SERVICE_URL}"

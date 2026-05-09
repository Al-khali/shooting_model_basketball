#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bootstrap.sh — ONE-TIME setup for Terraform state bucket + GCP project
#
# Run this ONCE before the first `terraform apply`.
# Prerequisites: gcloud CLI authenticated with Owner/Editor on billing account.
#
# Usage:
#   export PROJECT_ID="shoot-ai-dev"
#   export BILLING_ACCOUNT="XXXXXX-XXXXXX-XXXXXX"
#   export REGION="europe-west1"
#   bash infra/scripts/bootstrap.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
BILLING_ACCOUNT="${BILLING_ACCOUNT:?Set BILLING_ACCOUNT}"
REGION="${REGION:-europe-west1}"
TFSTATE_BUCKET="${PROJECT_ID}-tfstate"

echo "==> Bootstrap: project=${PROJECT_ID} region=${REGION}"

# 1. Create the GCP project (may already exist — ignore error)
gcloud projects create "${PROJECT_ID}" --quiet 2>/dev/null || echo "Project already exists — continuing"

# 2. Link billing account
gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}"

# 3. Enable Storage API (needed to create the tfstate bucket)
gcloud services enable storage.googleapis.com --project="${PROJECT_ID}"

# 4. Create Terraform remote state bucket (versioning + CMEK optional)
if ! gcloud storage buckets describe "gs://${TFSTATE_BUCKET}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud storage buckets create "gs://${TFSTATE_BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
  gcloud storage buckets update "gs://${TFSTATE_BUCKET}" --versioning
  echo "==> Created tfstate bucket: gs://${TFSTATE_BUCKET}"
else
  echo "==> tfstate bucket already exists: gs://${TFSTATE_BUCKET}"
fi

echo ""
echo "==> Bootstrap complete. Next steps:"
echo ""
echo "1. Init Terraform remote state:"
echo ""
echo "    cd infra/terraform"
echo "    terraform init \\"
echo "      -backend-config=\"bucket=${TFSTATE_BUCKET}\" \\"
echo "      -backend-config=\"prefix=terraform/state/${PROJECT_ID}\""
echo ""
echo "2. Import the pre-existing project (created by this script) into state:"
echo ""
echo "    terraform import google_project.shoot_ai ${PROJECT_ID}"
echo ""
echo "3. Apply with secrets via env vars:"
echo ""
echo "    export TF_VAR_gemini_api_key=\"\$GEMINI_API_KEY\""
echo "    export TF_VAR_api_keys='[\"dev-key-1\"]'   # list(string) format"
echo "    terraform apply -var-file=environments/dev/terraform.tfvars"

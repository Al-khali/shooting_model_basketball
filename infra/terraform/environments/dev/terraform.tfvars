# ──────────────────────────────────────────────────────────────────────────────
# Dev environment — Terraform variable values
# DO NOT commit real secrets here. Use terraform.tfvars.local or
# pass sensitive vars via: -var="gemini_api_key=$GEMINI_API_KEY"
# ──────────────────────────────────────────────────────────────────────────────

project_id      = "shoot-ai-poc"         # must match the GCP project you created
billing_account = "XXXXXX-XXXXXX-XXXXXX" # replace with your billing account ID
region          = "us-central1"          # Always Free tier region (Storage 5GB-month)
environment     = "dev"
app_name        = "shoot-ai"

# Cloud Run sizing — POC zero-budget: scale-to-zero + tight max
# Always Free tier covers ~180k GiB-seconds/month, easily enough for
# a POC (~100 analyses × 10s × 2Gi ≈ 2k GiB-seconds).
cloud_run_min_instances = 0 # scale-to-zero (essential for zero budget)
cloud_run_max_instances = 2 # cap blast radius if a test loop runs amok
cloud_run_cpu           = "1"
cloud_run_memory        = "2Gi"
cloud_run_timeout       = 300

# Secrets — override with env vars or tfvars.local:
#   export TF_VAR_gemini_api_key="your-key"
#   export TF_VAR_api_keys='["dev-key-1"]'

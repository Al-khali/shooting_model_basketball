variable "project_id" {
  description = "GCP project ID (must be globally unique, lowercase letters/digits/hyphens)"
  type        = string
}

variable "billing_account" {
  description = "GCP billing account ID (format: XXXXXX-XXXXXX-XXXXXX)"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and Artifact Registry"
  type        = string
  default     = "europe-west1"
}

variable "environment" {
  description = "Deployment environment (dev | staging | prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
  default     = "shoot-ai"
}

variable "cloud_run_min_instances" {
  description = "Minimum Cloud Run instances (0 = scale-to-zero)"
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum Cloud Run instances"
  type        = number
  default     = 3
}

variable "cloud_run_cpu" {
  description = "CPU allocation per Cloud Run instance"
  type        = string
  default     = "1"
}

variable "cloud_run_memory" {
  description = "Memory allocation per Cloud Run instance"
  type        = string
  default     = "2Gi"
}

variable "cloud_run_timeout" {
  description = "Request timeout in seconds (max 3600)"
  type        = number
  default     = 300
}

variable "gemini_api_key" {
  description = "Gemini API key — stored in Secret Manager, injected via Cloud Run secret ref"
  type        = string
  sensitive   = true
}

variable "api_keys" {
  description = "List of API keys for X-API-Key auth (e.g. [\"key1\",\"key2\"])"
  type        = list(string)
  sensitive   = true
  default     = []
}

variable "image_tag" {
  description = "Docker image tag to deploy (commit SHA from CI/CD, e.g. 'abc1234')"
  type        = string
  default     = "latest"
}

variable "github_repository" {
  description = "GitHub repository path for Workload Identity Federation (org/repo format)"
  type        = string
  default     = "Al-khali/shooting_model_basketball"
}

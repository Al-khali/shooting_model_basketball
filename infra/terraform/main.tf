terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Remote state — GCS bucket created by bootstrap.sh before first apply
  backend "gcs" {
    # bucket and prefix are set per environment via -backend-config flags in deploy.sh
    # Example: -backend-config="bucket=shoot-ai-tfstate-dev"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  name_prefix = "${var.app_name}-${var.environment}"

  common_labels = {
    app         = var.app_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

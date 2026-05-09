# ──────────────────────────────────────────────────────────────────────────────
# Cloud Run v2 Service
# ──────────────────────────────────────────────────────────────────────────────

locals {
  # Image is pushed by CI/CD with the commit SHA tag; placeholder for first apply
  image_url = "${var.region}-docker.pkg.dev/${var.project_id}/${local.name_prefix}/${var.app_name}"
}

resource "google_cloud_run_v2_service" "api" {
  project  = google_project.shoot_ai.project_id
  name     = local.name_prefix
  location = var.region
  labels   = local.common_labels

  # Allow unauthenticated requests — auth handled by X-API-Key middleware
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run.email

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = "${local.image_url}:latest"

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        cpu_idle          = true  # only charge CPU when processing requests
        startup_cpu_boost = true  # extra CPU during cold start
      }

      ports {
        container_port = 8080
      }

      # Secrets injected as environment variables via Secret Manager
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "API_KEYS"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_keys.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      # Liveness probe — Cloud Run will restart if /health returns non-2xx
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 15
        period_seconds        = 30
        failure_threshold     = 3
        timeout_seconds       = 10
      }

      # Startup probe — longer window during cold start / model loading
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 12  # up to 60s startup window
        timeout_seconds       = 10
      }
    }

    timeout = "${var.cloud_run_timeout}s"
  }

  depends_on = [
    google_project_service.apis["run.googleapis.com"],
    google_secret_manager_secret_version.gemini_api_key,
    google_secret_manager_secret_version.api_keys,
  ]
}

# Make service publicly accessible (auth via X-API-Key header, not Cloud Run IAM)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = google_project.shoot_ai.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

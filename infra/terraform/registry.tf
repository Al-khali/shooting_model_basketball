# ──────────────────────────────────────────────────────────────────────────────
# Artifact Registry — Docker repository
# ──────────────────────────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "docker" {
  project       = google_project.shoot_ai.project_id
  location      = var.region
  repository_id = local.name_prefix
  description   = "Docker images for ${var.app_name} (${var.environment})"
  format        = "DOCKER"
  labels        = local.common_labels

  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}

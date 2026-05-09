# ──────────────────────────────────────────────────────────────────────────────
# GCP Secret Manager — sensitive configuration
# ──────────────────────────────────────────────────────────────────────────────

resource "google_secret_manager_secret" "gemini_api_key" {
  project   = google_project.shoot_ai.project_id
  secret_id = "gemini-api-key"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "gemini_api_key" {
  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = var.gemini_api_key
}

resource "google_secret_manager_secret" "api_keys" {
  project   = google_project.shoot_ai.project_id
  secret_id = "api-keys"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "api_keys" {
  secret      = google_secret_manager_secret.api_keys.id
  secret_data = jsonencode(var.api_keys)
}

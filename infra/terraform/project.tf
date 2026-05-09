# ──────────────────────────────────────────────────────────────────────────────
# GCP Project + API enablement
# ──────────────────────────────────────────────────────────────────────────────

resource "google_project" "shoot_ai" {
  name            = local.name_prefix
  project_id      = var.project_id
  billing_account = var.billing_account
  labels          = local.common_labels
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "cloudbuild.googleapis.com",
    "iamcredentials.googleapis.com", # Workload Identity Federation
    "sts.googleapis.com",            # Security Token Service (WIF)
  ])

  project            = google_project.shoot_ai.project_id
  service            = each.value
  disable_on_destroy = false

  depends_on = [google_project.shoot_ai]
}

# ──────────────────────────────────────────────────────────────────────────────
# GCP Project + API enablement
#
# The project is created by bootstrap.sh so the GCS backend bucket can exist
# before the first `terraform apply`. After bootstrap, import it:
#
#   terraform import google_project.shoot_ai $PROJECT_ID
#
# This brings the existing project under Terraform management without
# recreating it (avoids "project already exists" error on first apply).
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

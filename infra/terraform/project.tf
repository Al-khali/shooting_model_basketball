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

  # `billing_account` is a one-time setup decision performed during
  # bootstrap.sh and persisted to state via `terraform import`. The repo's
  # default tfvars holds a placeholder ("XXXXXX-XXXXXX-XXXXXX") so the real
  # ID never lands in git — but every CI apply would then read that
  # placeholder and try to *swap* the project's billing account to it,
  # which (a) is wrong and (b) requires `cloudbilling.googleapis.com` to
  # call `setBillingAccount`. The CI service account doesn't drive the
  # billing lifecycle, the operator does (out-of-Terraform), so we mark
  # the field as ignored after import. The state still holds the real
  # account ID and Cloud Run keeps billing through it.
  lifecycle {
    ignore_changes = [billing_account]
  }
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "cloudbuild.googleapis.com",
    "iamcredentials.googleapis.com",       # Workload Identity Federation
    "sts.googleapis.com",                  # Security Token Service (WIF)
    "cloudresourcemanager.googleapis.com", # Required for the google_project data lookups Terraform performs on every plan/apply; without it the CI SA hits HTTP 403 during `terraform plan`
    "cloudbilling.googleapis.com",         # Required for any read/write on the project's billing link (even with ignore_changes, the provider still issues a Get on the billing association during refresh)
  ])

  project            = google_project.shoot_ai.project_id
  service            = each.value
  disable_on_destroy = false

  depends_on = [google_project.shoot_ai]
}

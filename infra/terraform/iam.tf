# ──────────────────────────────────────────────────────────────────────────────
# Service Accounts + IAM bindings
# ──────────────────────────────────────────────────────────────────────────────

# Service account used by the Cloud Run service at runtime
resource "google_service_account" "cloud_run" {
  project      = google_project.shoot_ai.project_id
  account_id   = "${local.name_prefix}-run"
  display_name = "Cloud Run SA — ${local.name_prefix}"

  depends_on = [google_project_service.apis["iam.googleapis.com"]]
}

# Grant secret read access
resource "google_secret_manager_secret_iam_member" "run_gemini_key" {
  project   = google_project.shoot_ai.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_api_keys" {
  project   = google_project.shoot_ai.project_id
  secret_id = google_secret_manager_secret.api_keys.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Workload Identity Federation — GitHub Actions → GCP (no JSON key)
# ──────────────────────────────────────────────────────────────────────────────

resource "google_iam_workload_identity_pool" "github" {
  project                   = google_project.shoot_ai.project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"
  description               = "WIF pool for GitHub Actions CI/CD"

  depends_on = [google_project_service.apis["iam.googleapis.com"]]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = google_project.shoot_ai.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "assertion.repository == 'Al-khali/shooting_model_basketball'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Service account for CI/CD (push images, deploy Cloud Run)
resource "google_service_account" "cicd" {
  project      = google_project.shoot_ai.project_id
  account_id   = "${local.name_prefix}-cicd"
  display_name = "CI/CD SA — ${local.name_prefix}"
}

# Allow GitHub Actions to impersonate the CI/CD SA via WIF
resource "google_service_account_iam_member" "cicd_wif" {
  service_account_id = google_service_account.cicd.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/Al-khali/shooting_model_basketball"
}

# CI/CD needs to push to Artifact Registry
resource "google_artifact_registry_repository_iam_member" "cicd_writer" {
  project    = google_project.shoot_ai.project_id
  location   = var.region
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cicd.email}"
}

# CI/CD needs to deploy Cloud Run
resource "google_project_iam_member" "cicd_run_developer" {
  project = google_project.shoot_ai.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

# CI/CD needs to pass the Cloud Run SA to the service
resource "google_service_account_iam_member" "cicd_act_as_run" {
  service_account_id = google_service_account.cloud_run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd.email}"
}

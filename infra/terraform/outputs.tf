output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${local.name_prefix}"
}

output "image_url" {
  description = "Full image URL (without tag)"
  value       = local.image_url
}

output "workload_identity_provider" {
  description = "WIF provider resource name — set as GCP_WORKLOAD_IDENTITY_PROVIDER secret in GitHub"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "cicd_service_account" {
  description = "CI/CD service account email — set as GCP_SERVICE_ACCOUNT secret in GitHub"
  value       = google_service_account.cicd.email
}

output "cloud_run_service_account" {
  description = "Cloud Run runtime service account email"
  value       = google_service_account.cloud_run.email
}

output "image" {
  value = local.image
}

output "app_url" {
  description = "Public URL to submit (chat UI)"
  value       = google_cloud_run_v2_service.app.uri
}

output "health_curl" {
  value = "curl -sS ${google_cloud_run_v2_service.app.uri}/health"
}

output "ask_curl" {
  value = "curl -sS -X POST ${google_cloud_run_v2_service.app.uri}/ask -H 'Content-Type: application/json' -d '{\"question\":\"What weapon does Usagi use?\"}'"
}

output "destroy_reminder" {
  value = "When grading is done: terraform destroy"
}

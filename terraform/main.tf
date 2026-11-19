# Assignment #3: deploy Chiikawa-pedia (RAG + remote LLM) with Terraform
#   1) create the resources  2) pull the application image  3) run the application (Cloud Run)

locals {
  services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
  ]
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.app.repository_id}/${var.name_prefix}-rag:${var.image_tag}"
}

resource "google_project_service" "services" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# Private registry for the application image
resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = "${var.name_prefix}-repo"
  format        = "DOCKER"
  depends_on    = [google_project_service.services]
}

# Identity the app runs as: may only read the OpenAI key secret
resource "google_service_account" "run" {
  account_id   = "${var.name_prefix}-run"
  display_name = "Chiikawa RAG Cloud Run service"
}

resource "google_secret_manager_secret_iam_member" "key_reader" {
  secret_id  = var.openai_secret_id
  role       = "roles/secretmanager.secretAccessor"
  member     = "serviceAccount:${google_service_account.run.email}"
  depends_on = [google_project_service.services]
}

# Pull the image and run it as a public HTTPS service
resource "google_cloud_run_v2_service" "app" {
  name                 = "${var.name_prefix}-rag"
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  deletion_protection  = false

  template {
    service_account = google_service_account.run.email
    timeout         = "120s"

    scaling {
      min_instance_count = 0 # scales to zero when idle: no cost between requests
      max_instance_count = 2
    }

    containers {
      image = local.image
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle = true
      }
      env {
        name = "OPENAI_API_KEY" # injected at runtime from Secret Manager, never in code or Terraform
        value_source {
          secret_key_ref {
            secret  = var.openai_secret_id
            version = "latest"
          }
        }
      }
      startup_probe {
        http_get {
          path = "/healthz"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 24
      }
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.key_reader]
}

# Public access: allow anyone to invoke the service
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

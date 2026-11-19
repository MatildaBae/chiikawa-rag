terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.20"
    }
  }
}

# Credentials come from `gcloud auth application-default login`; nothing secret lives in this repo.
provider "google" {
  project = var.project_id
  region  = var.region
}

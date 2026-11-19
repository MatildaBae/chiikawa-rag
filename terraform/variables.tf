variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for Artifact Registry and Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "chiikawa"
}

variable "image_tag" {
  description = "Image tag to deploy (bump it, e.g. v2, after pushing a new build)"
  type        = string
  default     = "v1"
}

variable "openai_secret_id" {
  description = "Secret Manager secret holding the OpenAI API key (created with gcloud, never stored in Terraform)"
  type        = string
  default     = "openai-api-key"
}

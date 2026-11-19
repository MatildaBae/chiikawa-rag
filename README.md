# Chiikawa-pedia: RAG app deployed on GCP with Terraform (Assignment #3)

A retrieval-augmented chatbot about the *Chiikawa* manga/anime (from my Lab 2 extra-credit notebook),
deployed as a public web app on **Google Cloud Run** with Terraform.

| Piece | What it does |
|---|---|
| `app/build_index.py` | Fetches ~50 Chiikawa fan-wiki pages + the Wikipedia article, chunks them (section-aware sentence windows), embeds with `bge-small-en-v1.5`, saves a FAISS index. Runs during `docker build`. |
| `app/rag.py` | Retrieval (FAISS cosine top-k + character-name boost) and grounded answers from `gpt-4o-mini` with `[n]` citations; refuses when nothing relevant is found. |
| `app/server.py` | Gradio chat UI at `/`, JSON API at `POST /ask`, health check at `GET /healthz`. |
| `terraform/` | Enables APIs, creates Artifact Registry, a service account, Secret Manager access, and the Cloud Run service that pulls and runs the image. |

**Secrets:** the OpenAI key lives only in Secret Manager and is injected at runtime. It is not in the code, the image, or the Terraform files.

## Deploy

```bash
# 0. variables
export PROJECT_ID="ieor4577-515217" REGION="us-central1"
gcloud config set project $PROJECT_ID

# 1. store the OpenAI key in Secret Manager (once)
gcloud services enable secretmanager.googleapis.com
read -s OPENAI_KEY && printf "%s" "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=- && unset OPENAI_KEY

# 2. create the registry
cd terraform
cp terraform.tfvars.example terraform.tfvars      # check project_id
terraform init
terraform apply -target=google_project_service.services -target=google_artifact_registry_repository.app

# 3. build (amd64 for Cloud Run) and push the image
cd ..
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker buildx build --platform linux/amd64 \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/chiikawa-repo/chiikawa-rag:v1 . --push

# 4. deploy: Cloud Run pulls the image and runs it
cd terraform
terraform apply
terraform output app_url
```

## Test

```bash
curl -sS https://<app_url>/healthz
curl -sS -X POST https://<app_url>/ask -H 'Content-Type: application/json' \
  -d '{"question":"What weapon does Usagi use?"}'
```
Or open `app_url` in a browser for the chat UI.

## Update the app
Rebuild with a new tag (e.g. `:v2`), then `terraform apply -var='image_tag=v2'`.

## Tear down
```bash
cd terraform && terraform destroy
gcloud secrets delete openai-api-key --quiet   # optional
```

*Data credits: Chiikawa Fan Wiki (Fandom) and Wikipedia, both CC BY-SA. Chiikawa © Nagano.*

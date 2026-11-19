# Chiikawa-pedia RAG: Gradio UI + JSON API on port 8080
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8080 MODEL_CACHE=/app/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build the knowledge base + FAISS index at image build time (downloads wiki pages and the
# embedding model once), so the container starts fast and needs no data at runtime.
COPY app/ app/
COPY data/ data/
RUN python -m app.build_index

EXPOSE 8080
CMD ["python", "-m", "app.server"]

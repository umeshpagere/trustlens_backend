# Use a full Python image for better compatibility
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Install FFmpeg and system libraries with robust retry logic
RUN for i in 1 2 3 4 5; do apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && exit 0 || sleep 10; done; exit 1

# Set working directory
WORKDIR /app

# Ensure model cache and temp storage exist and are writable
RUN mkdir -p /app/model_cache /tmp/trustlens && chmod -R 777 /app/model_cache /tmp/trustlens

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models to bake them into the image
COPY pre_download_models.py .
RUN python pre_download_models.py

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8080

# Start the application using Gunicorn (with Uvicorn worker for async support)
# We use 2 workers and 1 thread per worker to stay within Cloud Run memory limits (default 512MB-2GB)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120 run:app

# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system dependencies
# ffmpeg: required for video extraction
# libgl1: required for opencv-python-headless (even though headless, some libs might be needed)
# curl/git: useful for debugging or extra pulls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure the /tmp/trustlens directory exists and is writable
RUN mkdir -p /tmp/trustlens && chmod 777 /tmp/trustlens

# Expose the application port
EXPOSE $PORT

# Start the application using Hypercorn
# We use the 'run:app' directly
CMD ["hypercorn", "run:app", "--bind", "0.0.0.0:5000", "--keep-alive", "120"]

# Use a full Python image for better compatibility
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install FFmpeg and system libraries with robust retry logic
# Using libgl1 instead of libgl1-mesa-glx (often deprecated/renamed in newer Debian)
RUN for i in 1 2 3 4 5; do apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && exit 0 || sleep 10; done; exit 1

# Set working directory
WORKDIR /app

# Install Python dependencies in stages to stay within Render's memory limits
# First upgrade pip
RUN pip install --upgrade pip

# Install heavy ML libraries separately to maximize cache and manage memory
RUN pip install --no-cache-dir torch sentence-transformers

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

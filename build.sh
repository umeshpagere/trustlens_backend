#!/usr/bin/env bash
# Render Build Script

# 1. Install system dependencies (FFmpeg is required for video extraction)
echo "📦 [Build] Installing system dependencies..."
apt-get update && apt-get install -y ffmpeg

# 2. Install Python dependencies
echo "🐍 [Build] Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ [Build] Build process complete!"

import os
from app.main import create_app
from app.models.model_loader import load_embedding_model

# ----------------------------------------------------
# 1. Preload ML models once at startup
# ----------------------------------------------------
print("⚙️ [Startup] Preloading ML models...")
load_embedding_model()

# ----------------------------------------------------
# 2. Initialize Flask application
# ----------------------------------------------------
app = create_app()

# ----------------------------------------------------
# 3. Determine runtime port
# Render provides PORT environment variable
# ----------------------------------------------------
PORT = int(os.environ.get("PORT", 5000))

# ----------------------------------------------------
# 4. Local execution
# ----------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 TrustLens backend running on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
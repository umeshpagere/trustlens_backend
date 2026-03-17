import os
from app.main import create_app
import app.models.model_loader  # noqa: F401 — triggers model preload at import time

# ----------------------------------------------------
# 1. ML models preload automatically when model_loader is imported above
# ----------------------------------------------------

# ----------------------------------------------------
# 2. Initialize Flask application
# ----------------------------------------------------
app = create_app()

# ----------------------------------------------------
# 3. Determine runtime port
# Render provides PORT environment variable
# ----------------------------------------------------
PORT = int(os.environ.get("PORT", 8080))

# ----------------------------------------------------
# 4. Local execution
# ----------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 TrustLens backend running on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
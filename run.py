from app.main import create_app
from app.models.model_loader import load_embedding_model

# 1. Preload models once at startup
print("⚙️ [Startup] Preloading ML models...")
load_embedding_model()

# 2. Initialize Flask app
app = create_app()

# Expose 'app' for ASGI workers (Hypercorn / Uvicorn)
if __name__ == "__main__":
    from app.config.settings import Config
    print(f"🚀 TrustLens backend running locally on http://127.0.0.1:{Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)

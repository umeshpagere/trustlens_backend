from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys
import logging

from app.routes.analyze import analyze_bp
from app.config.settings import Config
from app.config.storage import init_temp_storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_app():
    # Initialize ephemeral storage for Render (/tmp)
    init_temp_storage()

    app = Flask(__name__)
    
    # Configure CORS for extension compatibility
    # In production, Cloud Run instances need to handle preflight requests correctly
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    @app.route('/')
    def index():
        return jsonify({
            "status": "online",
            "message": "TrustLens Backend API is running on Cloud Run",
            "version": "1.2.1"
        }), 200

    
    app.register_blueprint(analyze_bp, url_prefix='/api/analyze')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status": "healthy", "service": "trustlens-backend"}), 200

    @app.route('/health/models')
    def health_models():
        """Probe each major dependency and report status. Returns 503 if any are degraded."""
        from app.models.model_loader import embedding_model
        from app.services.evidence_pipeline.evidence_filter import get_evidence_filter
        from app.config.settings import Config
        from app.config.azure import get_azure_client
        import pymongo

        results = {}
        all_ok = True

        # 1. Embedding model
        try:
            if embedding_model is None:
                raise RuntimeError("embedding_model is None")
            # Quick sanity encode
            embedding_model.encode(["ping"])
            results["embedding_model"] = "loaded"
        except Exception as em_err:
            results["embedding_model"] = f"error: {em_err}"
            all_ok = False

        # 2. NLI evidence filter model
        try:
            nli = get_evidence_filter()
            if nli is None:
                raise RuntimeError("NLI model is None")
            results["nli_model"] = "loaded"
        except Exception as nli_err:
            results["nli_model"] = f"error: {nli_err}"
            all_ok = False

        # 3. Azure OpenAI (lightweight connectivity probe)
        try:
            client = get_azure_client()
            if client is None:
                raise RuntimeError("Azure client returned None")
            results["azure_openai"] = "connected"
        except Exception as az_err:
            results["azure_openai"] = f"error: {az_err}"
            all_ok = False

        # 4. MongoDB connectivity probe
        try:
            mongo_uri = Config.MONGODB_URI
            if not mongo_uri:
                raise RuntimeError("MONGODB_URI not set")
            client_probe = pymongo.MongoClient(
                mongo_uri, serverSelectionTimeoutMS=3000, tls=True,
                tlsCAFile=__import__('certifi').where()
            )
            # Ping the server
            client_probe[Config.MONGODB_DATABASE].command("ping")
            client_probe.close()
            results["mongodb"] = "connected"
        except Exception as mdb_err:
            results["mongodb"] = f"error: {mdb_err}"
            all_ok = False

        status_code = 200 if all_ok else 503
        return jsonify({
            "status": "ok" if all_ok else "degraded",
            **results
        }), status_code

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "message": "Route not found"
        }), 404
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        logging.error(f"Unhandled Exception: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "Internal Server Error"
        }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    print(f"🚀 TrustLens backend running on http://127.0.0.1:{Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)

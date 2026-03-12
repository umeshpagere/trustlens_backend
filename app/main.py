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
    
    # Configure CORS for production (replace with actual frontend domain in production)
    # For now, allowing localhost for development and a placeholder for production
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://trustlens-frontend.onrender.com").split(",")
    CORS(app, origins=allowed_origins)
    
    app.register_blueprint(analyze_bp, url_prefix='/api/analyze')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status": "healthy", "service": "trustlens-backend"}), 200

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

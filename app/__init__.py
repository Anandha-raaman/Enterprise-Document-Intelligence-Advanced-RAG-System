from flask import Flask, render_template
from flask_cors import CORS
from app.config import Config
from app.api.routes import api_bp

def create_app(config_class=Config):
    """Flask application factory."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates"
    )
    app.config.from_object(config_class)

    # Enable CORS for cross-origin API integration
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API Blueprint
    app.register_blueprint(api_bp)

    # Web Dashboard Route
    @app.route("/")
    def index():
        return render_template("index.html")

    return app

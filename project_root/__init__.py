from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # Enable CORS (replace with your actual frontend URL if needed)
    CORS(app)

    # Register the routes
    from .routes import main
    app.register_blueprint(main)

    return app

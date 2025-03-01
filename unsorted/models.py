# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(500))
    ai_response = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=db.func.now())

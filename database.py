import os
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from datetime import datetime

db = SQLAlchemy()

def init_db(app: Flask):
    database_url = os.environ.get('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.String(500), nullable=False)
    answer_text = db.Column(db.String(500), nullable=False)
    question_type = db.Column(db.String(50), default='multiple_choice')  # multiple_choice, fill_blank, short_answer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    source_text = db.Column(db.Text)  # Original text that generated this question
    
    def to_dict(self):
        return {
            'id': self.id,
            'question_text': self.question_text,
            'answer_text': self.answer_text,
            'question_type': self.question_type,
            'created_at': self.created_at.isoformat(),
            'source_text': self.source_text
        } 
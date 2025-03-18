import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from collector import wikipedia_collector
from analyzer import question_generator
from dotenv import load_dotenv
import nltk

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///exams.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(200), nullable=False)
    questions = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_exam():
    data = request.get_json()
    topic = data.get('topic')
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    try:
        # Fetch Wikipedia content
        wiki_content = wikipedia_collector.get_topic_content(topic)
        if not wiki_content:
            return jsonify({'error': 'Could not find Wikipedia content for the given topic'}), 404
        
        # Process content into chunks suitable for question generation
        text_chunks = wikipedia_collector.process_content(wiki_content)
        
        # Generate questions
        questions = question_generator.generate_questions(text_chunks)
        
        # Save to database
        exam = Exam(
            topic=topic,
            questions=questions
        )
        db.session.add(exam)
        db.session.commit()
        
        return jsonify({
            'id': exam.id,
            'topic': exam.topic,
            'questions': exam.questions
        })
        
    except Exception as e:
        app.logger.error(f"Error generating exam: {str(e)}")
        return jsonify({'error': 'Failed to generate exam'}), 500

@app.route('/api/exams/<int:exam_id>', methods=['GET'])
def get_exam(exam_id):
    exam = Exam.query.get_or_404(exam_id)
    return jsonify({
        'id': exam.id,
        'topic': exam.topic,
        'questions': exam.questions,
        'created_at': exam.created_at.isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 
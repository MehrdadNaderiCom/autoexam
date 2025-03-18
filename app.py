import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from collector import wikipedia_collector
from question_generator import question_generator
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
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://') or 'sqlite:///exam.db'
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

@app.route('/generate', methods=['POST'])
def generate_exam():
    data = request.get_json()
    topic = data.get('topic')
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    try:
        # Fetch content from Wikipedia
        content = wikipedia_collector.get_topic_content(topic)
        if not content:
            return jsonify({'error': 'Could not find content for the given topic'}), 404
        
        # Process the content
        processed_content = wikipedia_collector.process_content(content)
        
        # Generate questions
        questions = question_generator.generate_questions(processed_content)
        
        # Save to database
        exam = Exam(topic=topic, questions=questions)
        db.session.add(exam)
        db.session.commit()
        
        return jsonify({
            'id': exam.id,
            'topic': exam.topic,
            'questions': exam.questions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/exam/<int:exam_id>', methods=['GET'])
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
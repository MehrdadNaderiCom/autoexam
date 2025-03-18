import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from collector import WikipediaCollector
from question_generator import QuestionGenerator
from dotenv import load_dotenv
import nltk
import wikipedia

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://') or 'sqlite:///exam.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def download_nltk_data():
    """Download NLTK data with error handling."""
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    required_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True, download_dir=nltk_data_dir)
            logger.info(f"Successfully downloaded NLTK package: {package}")
        except Exception as e:
            logger.warning(f"Failed to download NLTK package {package}: {str(e)}")
            # Continue even if download fails - we'll handle missing data in the question generator

# Try to download NLTK data, but continue if it fails
try:
    download_nltk_data()
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {str(e)}")

try:
    # Initialize question generator and collector
    question_generator = QuestionGenerator()
    wikipedia_collector = WikipediaCollector()
    logger.info("Successfully initialized QuestionGenerator and WikipediaCollector")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    # Continue anyway, we'll handle errors in the endpoints

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(200), nullable=False)
    questions = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(200), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    options = db.Column(db.Text, nullable=False)  # Stored as JSON string
    answer = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'topic': self.topic,
            'question': self.question_text,
            'options': eval(self.options),  # Convert JSON string back to list
            'answer': self.answer,
            'created_at': self.created_at.isoformat()
        }

# Create database tables
def init_db():
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_exam():
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        logger.info(f"Fetching content for topic: {topic}")
        content = wikipedia_collector.get_topic_content(topic)
        if not content:
            return jsonify({'error': 'Could not find content for the given topic'}), 404
        
        logger.info("Processing content")
        processed_content = wikipedia_collector.process_content(content)
        
        logger.info("Generating questions")
        questions = question_generator.generate_questions(processed_content)
        
        if not questions:
            return jsonify({'error': 'Could not generate questions from the content'}), 422
        
        logger.info("Saving to database")
        exam = Exam(topic=topic, questions=questions)
        db.session.add(exam)
        db.session.commit()
        
        # Format questions for frontend
        formatted_questions = []
        for q in questions:
            if q['type'] == 'multiple_choice':
                formatted_questions.append({
                    'question_type': 'multiple_choice',
                    'question_text': q['question'],
                    'options': q['options'],
                    'answer_text': q['answer']
                })
            else:  # fill_blank
                formatted_questions.append({
                    'question_type': 'fill_blank',
                    'question_text': q['question'],
                    'answer_text': q['answer']
                })
        
        return jsonify({
            'id': exam.id,
            'topic': exam.topic,
            'questions': formatted_questions
        })
    
    except Exception as e:
        logger.error(f"Error in generate_exam: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/exam/<int:exam_id>', methods=['GET'])
def get_exam(exam_id):
    try:
        exam = Exam.query.get_or_404(exam_id)
        return jsonify({
            'id': exam.id,
            'topic': exam.topic,
            'questions': exam.questions,
            'created_at': exam.created_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Error in get_exam: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate_questions():
    """Generate questions from Wikipedia content."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        num_questions = int(data.get('num_questions', 5))
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Search Wikipedia
        try:
            # First try to get the exact page
            page = wikipedia.page(topic)
        except wikipedia.exceptions.DisambiguationError as e:
            # If disambiguation error, use the first option
            page = wikipedia.page(e.options[0])
        except wikipedia.exceptions.PageError:
            return jsonify({'error': 'Topic not found on Wikipedia'}), 404
        
        # Generate questions
        questions = question_generator.generate_questions(page.content, num_questions)
        
        # Store questions in database
        for q in questions:
            question = Question(
                topic=topic,
                question_text=q['question'],
                options=str(q['options']),  # Convert list to string for storage
                answer=q['answer']
            )
            db.session.add(question)
        
        db.session.commit()
        
        return jsonify({
            'questions': questions,
            'topic': topic,
            'wikipedia_url': page.url
        })
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get all previously generated questions."""
    try:
        questions = Question.query.order_by(Question.created_at.desc()).all()
        return jsonify({
            'questions': [q.to_dict() for q in questions]
        })
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<int:question_id>', methods=['DELETE'])
def delete_question(question_id):
    """Delete a specific question."""
    try:
        question = Question.query.get_or_404(question_id)
        db.session.delete(question)
        db.session.commit()
        return jsonify({'message': 'Question deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting question: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize database when the app starts
init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
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
    """Model for storing individual questions."""
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(200), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), nullable=False)  # 'multiple_choice' or 'fill_blank'
    options = db.Column(db.Text)  # Stored as JSON string
    answer = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text)
    source_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert question to dictionary."""
        return {
            'id': self.id,
            'topic': self.topic,
            'question': self.question_text,
            'type': self.question_type,
            'options': eval(self.options) if self.options else [],
            'answer': self.answer,
            'explanation': self.explanation,
            'source_url': self.source_url,
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
def index():
    return render_template('index.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history')
def get_history():
    try:
        exams = Exam.query.order_by(Exam.created_at.desc()).all()
        return jsonify([{
            'id': exam.id,
            'topic': exam.topic,
            'questions': exam.questions,
            'created_at': exam.created_at.isoformat()
        } for exam in exams])
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/generate_exam', methods=['POST'])
def generate_exam():
    """Generate exam questions based on the given topic."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        num_questions = int(data.get('num_questions', 5))
        
        logger.info(f"Received request to generate {num_questions} questions for topic: {topic}")

        if not topic:
            logger.warning("No topic provided in request")
            return jsonify({'error': 'Topic is required'}), 400

        # Get content from Wikipedia
        logger.info("Fetching content from Wikipedia")
        content = wikipedia_collector.get_topic_content(topic)
        if not content:
            logger.warning(f"No Wikipedia content found for topic: {topic}")
            return jsonify({'error': 'Could not find content for the given topic'}), 404

        # Generate questions
        logger.info("Generating questions")
        questions = question_generator.generate_questions(content, num_questions)
        if not questions:
            logger.warning("Failed to generate questions")
            return jsonify({'error': 'Could not generate questions'}), 500

        # Store questions in the database
        stored_questions = []
        for q in questions:
            question = Question(
                topic=topic,
                question_text=q['question'],
                question_type=q['type'],
                options=str(q.get('options', [])),
                answer=q['answer'],
                explanation=q.get('explanation', ''),
                source_url=q.get('source_url', '')
            )
            db.session.add(question)
            stored_questions.append(question.to_dict())

        # Store the exam
        exam = Exam(
            topic=topic,
            questions=stored_questions
        )
        db.session.add(exam)
        
        try:
            db.session.commit()
            logger.info("Successfully stored questions and exam in database")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to store questions in database: {str(e)}")
            # Continue anyway - we can still return the generated questions

        return jsonify({
            'questions': stored_questions,
            'exam_id': exam.id if exam else None
        })

    except Exception as e:
        logger.error(f"Error in generate_exam: {str(e)}")
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
import os
from flask import Flask, request, jsonify, render_template
from database import db, init_db, Question
from analyzer import question_generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
init_db(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    max_questions = data.get('max_questions', 5)
    
    try:
        # Generate questions using the question generator
        questions = question_generator.generate_questions(text, max_questions)
        
        # Save questions to database
        saved_questions = []
        for q in questions:
            question = Question(
                question_text=q['question_text'],
                answer_text=q['answer_text'],
                question_type=q['question_type'],
                source_text=q['source_text']
            )
            db.session.add(question)
            saved_questions.append(question.to_dict())
        
        db.session.commit()
        return jsonify({'questions': saved_questions})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/questions', methods=['GET'])
def list_questions():
    questions = Question.query.order_by(Question.created_at.desc()).all()
    return jsonify({'questions': [q.to_dict() for q in questions]})

@app.route('/questions/<int:question_id>', methods=['DELETE'])
def delete_question(question_id):
    question = Question.query.get_or_404(question_id)
    try:
        db.session.delete(question)
        db.session.commit()
        return '', 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 
import pytest
from app import app
from database import db, Question

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_generate_questions_no_text(client):
    response = client.post('/generate', json={})
    assert response.status_code == 400
    assert 'error' in response.json

def test_generate_questions(client):
    test_text = "This is a test sentence for question generation."
    response = client.post('/generate', json={'text': test_text})
    assert response.status_code == 200
    assert 'questions' in response.json
    
def test_list_questions(client):
    # Add a test question
    question = Question(
        question_text='Test question?',
        answer_text='Test answer',
        question_type='multiple_choice',
        source_text='Test source'
    )
    with app.app_context():
        db.session.add(question)
        db.session.commit()
    
    response = client.get('/questions')
    assert response.status_code == 200
    assert 'questions' in response.json
    assert len(response.json['questions']) > 0 
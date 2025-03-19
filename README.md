# AutoExam Generator

An AI-powered exam question generator that automatically creates educational questions from Wikipedia articles using natural language processing and OpenAI's GPT-3.5.

## Project Overview

AutoExam is a web application that automates the creation of exam questions from educational texts. It combines Wikipedia content retrieval, natural language processing, and AI-powered question generation to create high-quality multiple-choice and fill-in-the-blank questions.

### Key Features

- **Automated Question Generation**: Creates multiple-choice questions from any Wikipedia topic
- **AI Enhancement**: Uses OpenAI's GPT-3.5 to generate high-quality questions and distractors
- **Content Processing**: Intelligent content extraction and processing from Wikipedia
- **Question History**: Tracks and stores generated questions for future reference
- **RESTful API**: Complete API for question generation and management
- **Database Integration**: Persistent storage of questions and exams
- **Error Handling**: Robust error handling for API and content processing
- **Responsive UI**: Clean and intuitive web interface

## Architecture

The application is built with a modular architecture:

- **Web Layer**: Flask web framework with REST API endpoints
- **Data Layer**: SQLAlchemy ORM with PostgreSQL/SQLite support
- **Content Collection**: Wikipedia API integration for content retrieval
- **Question Generation**: NLTK for text processing + OpenAI GPT-3.5 for enhancement
- **Frontend**: Bootstrap-based responsive design

## Technical Stack

- **Backend**: Python 3.9+
- **Web Framework**: Flask 2.0.1
- **Database**: SQLAlchemy + PostgreSQL/SQLite
- **NLP**: NLTK 3.8.1
- **AI**: OpenAI GPT-3.5
- **Content**: Wikipedia-API
- **Frontend**: HTML5, Bootstrap, JavaScript

## Setup and Installation

1. Clone the repository
2. Install Python 3.9 or higher
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   DATABASE_URL=your_database_url (optional)
   ```
5. Initialize the database:
   ```bash
   flask db upgrade
   ```
6. Run the application:
   ```bash
   python app.py
   ```
7. Access the application at `http://localhost:5000`

## API Endpoints

- `POST /generate_exam`: Generate questions for a topic
- `GET /api/history`: Retrieve question generation history
- `GET /exam/<id>`: Get a specific exam by ID
- `DELETE /delete/<id>`: Delete a specific question
- `GET /health`: Application health check

## Usage Guide

1. **Topic Selection**:

   - Enter any educational topic in the input field
   - Examples: "Python programming", "World War II", "Quantum Physics"

2. **Question Generation**:

   - Select the desired number of questions (default: 5)
   - Click "Generate Questions"
   - Wait for AI-enhanced questions to be generated

3. **Results**:

   - View generated multiple-choice questions
   - Each question includes:
     - Question text
     - Four answer options
     - Correct answer
     - Explanation
     - Source reference

4. **History**:
   - Access previously generated questions
   - View and manage question history
   - Delete unwanted questions

## Error Handling

The application includes comprehensive error handling for:

- Wikipedia content retrieval
- OpenAI API integration
- Database operations
- Invalid user input
- Network issues

## Development Notes

- Uses the legacy OpenAI API (v0.28.0) for stable question generation
- Implements fallback mechanisms for NLTK data availability
- Includes detailed logging for debugging and monitoring
- Supports both development and production configurations

## Production Deployment

The application is configured for deployment on Heroku:

- Automatic database URL handling
- Gunicorn WSGI server support
- Environment variable configuration
- PostgreSQL database support

## License

This project is part of an academic assignment and is provided as-is for educational purposes.

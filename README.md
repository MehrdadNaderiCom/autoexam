# AutoExam Generator

An AI-powered exam question generator that creates multiple-choice questions from Wikipedia articles.

## Features

- Generate multiple-choice questions from any Wikipedia topic
- Customizable number of questions
- Question history tracking
- Clean and intuitive user interface
- Built with Flask and NLTK

## Setup

1. Install Python 3.9 or higher
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter a topic in the input field (e.g., "Python programming", "World War II", "Quantum Physics")
2. Select the number of questions you want to generate
3. Click "Generate Questions"
4. View and answer the generated questions
5. Access your question history below

## Technical Details

- Built with Flask web framework
- Uses NLTK for natural language processing
- SQLite database for storing questions
- Bootstrap for responsive design

## Requirements

- Python 3.9+
- Flask
- NLTK
- Wikipedia API
- Other dependencies listed in requirements.txt

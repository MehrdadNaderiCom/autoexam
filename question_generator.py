import os
import random
import nltk
import logging
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator."""
        self.use_basic_tokenization = True
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/averaged_perceptron_tagger')
            self.use_basic_tokenization = False
            logger.info("NLTK data found, using advanced tokenization")
        except LookupError:
            logger.info("NLTK data not found, using basic tokenization")

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Split on common sentence endings
        text = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|')
        sentences = text.split('|')
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into sentences using either NLTK or basic tokenization."""
        try:
            if not self.use_basic_tokenization:
                return nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
        return self._basic_tokenize(text)

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract potential keywords from a sentence."""
        try:
            if not self.use_basic_tokenization:
                words = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(words)
                return [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        except Exception as e:
            logger.warning(f"NLTK keyword extraction failed: {str(e)}")
        
        # Fallback: return words with 4 or more characters
        words = sentence.split()
        return [word for word in words if len(word) >= 4 and word.isalnum()]

    def generate_questions(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate a list of questions from the given text."""
        questions = []
        sentences = self._tokenize_text(text)
        
        if not sentences:
            return questions

        # Try to generate twice as many questions as requested
        attempts = min(len(sentences), num_questions * 2)
        used_sentences = set()

        while len(questions) < num_questions and attempts > 0:
            sentence = random.choice(sentences)
            if sentence in used_sentences:
                attempts -= 1
                continue

            used_sentences.add(sentence)
            keywords = self._extract_keywords(sentence)

            if keywords:
                keyword = random.choice(keywords)
                question = self._generate_question(sentence, keyword)
                if question:
                    questions.append(question)

            attempts -= 1

        return questions[:num_questions]

    def _generate_question(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a question from a sentence and keyword."""
        # Create a fill-in-the-blank question
        question_text = sentence.replace(keyword, "________")
        
        return {
            'type': 'fill_blank',
            'question': question_text,
            'answer': keyword,
            'options': []  # No options for fill-in-the-blank
        }

# Initialize the question generator
question_generator = QuestionGenerator() 
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
        logger.info("QuestionGenerator initialized with basic tokenization")

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Split on common sentence endings
        text = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|')
        sentences = text.split('|')
        return [s.strip() for s in sentences if s.strip()]

    def _generate_question_from_text(self, sentence: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a question-answer pair from the given sentence."""
        try:
            # Simple word-based approach
            words = sentence.split()
            
            # Find important words (longer words are often more important)
            important_words = [w for w in words if len(w) > 4]
            
            if not important_words:
                return None
            
            # Choose a random important word
            target_word = random.choice(important_words)
            
            # Generate question templates
            templates = [
                f"What is the significance of '{target_word}' in this context?",
                f"Can you explain what '{target_word}' means here?",
                f"What role does '{target_word}' play in this statement?"
            ]
            
            question = random.choice(templates)
            
            # Generate distractors
            distractors = []
            other_words = [w for w in important_words if w != target_word]
            
            if other_words:
                distractors.extend(random.sample(other_words, min(2, len(other_words))))
            
            while len(distractors) < 3:
                distractor = f"Not related to {target_word}"
                if distractor not in distractors:
                    distractors.append(distractor)
            
            options = distractors + [sentence]
            random.shuffle(options)
            
            return {
                'type': 'multiple_choice',
                'question': question,
                'options': options,
                'answer': sentence
            }
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return None

    def generate_questions(self, content: str, num_questions: int = 5) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Generate questions from the given content.
        
        Args:
            content (str): The text content to generate questions from
            num_questions (int): Number of questions to generate
            
        Returns:
            List[Dict]: List of questions with their answers
        """
        # Split content into sentences
        sentences = self._basic_tokenize(content)
        
        # Filter out short sentences and those that might not be informative
        valid_sentences = [
            sent for sent in sentences 
            if len(sent.split()) > 8 and  # Minimum word count
            not any(x in sent.lower() for x in ['copyright', 'all rights reserved', 'http', 'www'])
        ]
        
        # Generate questions
        questions = []
        attempts = 0
        max_attempts = num_questions * 2  # Allow for some failures
        
        while len(questions) < num_questions and attempts < max_attempts and valid_sentences:
            # Select a random sentence that hasn't been used yet
            sentence = random.choice(valid_sentences)
            valid_sentences.remove(sentence)  # Avoid reusing the same sentence
            
            question_data = self._generate_question_from_text(sentence)
            if question_data:
                questions.append(question_data)
            
            attempts += 1
        
        return questions

# Initialize the question generator
question_generator = QuestionGenerator() 
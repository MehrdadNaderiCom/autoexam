import os
import random
import nltk
import logging
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with required data."""
        # Set up NLTK
        self.nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(self.nltk_data_dir):
            os.makedirs(self.nltk_data_dir, exist_ok=True)
        nltk.data.path.append(self.nltk_data_dir)
        
        # Try to ensure NLTK data is available
        try:
            self._ensure_nltk_data()
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}. Will use basic tokenization.")
            self.use_basic_tokenization = True
        else:
            self.use_basic_tokenization = False
        
        logger.info("QuestionGenerator initialized successfully")

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
                logger.info(f"Found existing {package} data")
            except LookupError:
                logger.info(f"Downloading {package}...")
                try:
                    nltk.download(package, quiet=True, download_dir=self.nltk_data_dir)
                    logger.info(f"Successfully downloaded {package}")
                except Exception as e:
                    logger.warning(f"Failed to download {package}: {e}")
                    raise

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic sentence tokenization when NLTK data is not available."""
        sentences = []
        current = []
        
        for word in text.split():
            current.append(word)
            if word.endswith(('.', '!', '?')) and len(word) > 1:
                sentences.append(' '.join(current))
                current = []
        
        if current:  # Add any remaining text
            sentences.append(' '.join(current))
        
        return sentences

    def _generate_question_from_text(self, sentence: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a question-answer pair from the given sentence."""
        try:
            # Tokenize and tag parts of speech
            tokens = nltk.word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            # Find important words (nouns, verbs, adjectives)
            important_words = []
            for word, tag in tagged:
                if tag.startswith(('NN', 'VB', 'JJ')):  # Nouns, Verbs, Adjectives
                    important_words.append(word)
            
            if not important_words:
                return None
            
            # Choose a random important word
            target_word = random.choice(important_words)
            
            # Generate question based on word type
            word_type = next(tag for word, tag in tagged if word == target_word)
            
            question_templates = {
                'NN': [  # Nouns
                    f"What is {target_word}?",
                    f"Can you explain the concept of {target_word}?",
                    f"What role does {target_word} play in this context?"
                ],
                'VB': [  # Verbs
                    f"What does {target_word} mean in this context?",
                    f"How does {target_word} relate to the main topic?",
                    f"Can you explain the process of {target_word}?"
                ],
                'JJ': [  # Adjectives
                    f"What makes something {target_word}?",
                    f"How does {target_word} affect the context?",
                    f"Can you explain the significance of being {target_word}?"
                ]
            }
            
            # Get appropriate templates based on word type
            templates = question_templates.get(word_type[:2], question_templates['NN'])
            question = random.choice(templates)
            
            # Generate distractors
            distractors = []
            other_words = [word for word in important_words if word != target_word]
            
            if other_words:
                distractors.extend(random.sample(other_words, min(2, len(other_words))))
            
            while len(distractors) < 3:
                # Generate plausible distractors
                if word_type.startswith('NN'):
                    distractors.append(f"Different type of {target_word}")
                elif word_type.startswith('VB'):
                    distractors.append(f"Alternative to {target_word}")
                else:
                    distractors.append(f"Opposite of {target_word}")
            
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
        if self.use_basic_tokenization:
            sentences = self._basic_tokenize(content)
        else:
            try:
                sentences = sent_tokenize(content)
            except Exception as e:
                logger.warning(f"Error using NLTK tokenizer: {e}. Falling back to basic tokenization.")
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
import os
import random
import nltk
import logging
import spacy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
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
        self._ensure_nltk_data()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
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
                nltk.download(package, quiet=True)
                logger.info(f"Successfully downloaded {package}")

    def _generate_question_from_text(self, sentence: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a question-answer pair from the given sentence."""
        try:
            doc = self.nlp(sentence)
            
            # Extract key information
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            if not entities:
                return None
                
            # Choose a random entity to ask about
            entity, entity_type = random.choice(entities)
            
            # Generate question based on entity type
            question_templates = {
                'PERSON': [
                    f"Who is {entity}?",
                    f"What role did {entity} play in this context?",
                ],
                'ORG': [
                    f"What is {entity}?",
                    f"What is the significance of {entity}?",
                ],
                'DATE': [
                    f"When did this event occur?",
                    f"What happened in {entity}?",
                ],
                'GPE': [
                    f"What is significant about {entity}?",
                    f"Where is {entity} located?",
                ],
                'DEFAULT': [
                    f"What is {entity}?",
                    f"Can you explain the significance of {entity}?",
                ]
            }
            
            templates = question_templates.get(entity_type, question_templates['DEFAULT'])
            question = random.choice(templates)
            
            # Generate distractors
            other_entities = [e[0] for e in entities if e[0] != entity]
            distractors = []
            
            if other_entities:
                distractors.extend(random.sample(other_entities, min(2, len(other_entities))))
            
            while len(distractors) < 3:
                # Generate a plausible but incorrect answer
                distractor = f"Not {entity}" if len(distractors) == 0 else f"Something related to {entity}"
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
        sentences = sent_tokenize(content)
        
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
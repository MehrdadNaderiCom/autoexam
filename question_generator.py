import os
import random
import nltk
import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with required models and data."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize T5 model and tokenizer
        logger.info("Loading T5 model and tokenizer...")
        self.model_name = 'google/flan-t5-base'  # Using FLAN-T5 for better question generation
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.info("Model loaded successfully")
        
        # Set up NLTK
        self.nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(self.nltk_data_dir):
            os.makedirs(self.nltk_data_dir, exist_ok=True)
        nltk.data.path.append(self.nltk_data_dir)
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("Found existing punkt data")
        except LookupError:
            logger.info("Downloading punkt...")
            nltk.download('punkt', quiet=True)
            logger.info("Successfully downloaded punkt")

    def _generate_question_from_text(self, context: str, max_length: int = 64) -> Dict[str, str]:
        """Generate a question-answer pair from the given context using T5."""
        try:
            # Prepare input text
            input_text = f"generate question: {context}"
            
            # Tokenize and generate question
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    length_penalty=1.5,
                    early_stopping=True
                )
            
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Generate answer
            answer_input = f"answer question: {question} context: {context}"
            answer_inputs = self.tokenizer(answer_input, return_tensors="pt", max_length=512, truncation=True)
            answer_inputs = {k: v.to(self.device) for k, v in answer_inputs.items()}
            
            with torch.no_grad():
                answer_outputs = self.model.generate(
                    **answer_inputs,
                    max_length=64,
                    num_beams=4,
                    length_penalty=1.5,
                    early_stopping=True
                )
            
            answer = self.tokenizer.decode(answer_outputs[0], skip_special_tokens=True)
            
            # Generate distractors for multiple choice
            distractors_input = f"generate incorrect options for question: {question} correct answer: {answer}"
            distractors_inputs = self.tokenizer(distractors_input, return_tensors="pt", max_length=512, truncation=True)
            distractors_inputs = {k: v.to(self.device) for k, v in distractors_inputs.items()}
            
            with torch.no_grad():
                distractors_outputs = self.model.generate(
                    **distractors_inputs,
                    max_length=128,
                    num_beams=4,
                    num_return_sequences=3,
                    length_penalty=1.5,
                    early_stopping=True
                )
            
            distractors = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in distractors_outputs
            ]
            
            options = distractors + [answer]
            random.shuffle(options)
            
            return {
                'type': 'multiple_choice',
                'question': question,
                'options': options,
                'answer': answer
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
        # Split content into sentences and select the most informative ones
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